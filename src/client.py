import openai
import pydantic
from .config import get_api_key
import json
import pandas as pd
from io import StringIO
import time
from uuid import uuid4
from openai.lib._pydantic import to_strict_json_schema
import glob
from concurrent.futures import ThreadPoolExecutor
import random
from tqdm import tqdm

# Assuming these are in the same directory structure as before
from src.prompts import create_xml_to_csv_script_prompt, create_sql_prompt, create_chemistry_prompt, create_metadata_prompt
from src.formats import Script, SQL, RowChemistry, Metadata
from src.sql import get_sql_conn, add_secondary_sql_table
from src.utils import read_xml, get_csv_from_response

class ASOClient:
    """
    An updated client for OpenAI API calls, specifically tailored for gpt-5 models
    and their new parameters like verbosity, reasoning, flex processing, and custom tools.
    """
    def __init__(self, 
                 model: str = "gpt-5", 
                 verbosity: str = "low",
                 reasoning_effort: str = "medium",
                 service_tier: str = None,
                 timeout: float = None):
        """
        Initializes the client.

        Args:
            model (str): The model to use, e.g., "gpt-5", "gpt-5-mini".
            verbosity (str): Controls output expansiveness. One of "low", "medium", "high".
            reasoning_effort (str): Controls reasoning tokens. One of "minimal", "medium", "high".
            service_tier (str): Service tier for processing. Use "flex" for cost optimization with slower response times.
            timeout (float): Request timeout in seconds. Auto-adjusts to 900s (15 min) for flex processing if not specified.
        """

        self.api_key = get_api_key('openai')
        base_url = "https://api.openai.com/v1"
        
        # Auto-adjust timeout for flex processing
        if timeout is None:
            if service_tier == "flex":
                timeout = 900.0  # 15 minutes for flex processing
            else:
                timeout = 600.0  # 10 minutes default
        
        self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.verbosity = verbosity
        self.reasoning_effort = reasoning_effort
        self.service_tier = service_tier
        self.timeout = timeout
        self.usage = {}

    def _handle_api_call_with_retry(self, api_call_func, max_retries: int = 3):
        """
        Handle API calls with retry logic for flex processing resource unavailability.
        
        Args:
            api_call_func: Function that makes the API call
            max_retries: Maximum number of retries for 429 errors
            
        Returns:
            API response
        """
        for attempt in range(max_retries + 1):
            try:
                return api_call_func()
            except openai.APIError as e:
                if e.status_code == 429 and "Resource Unavailable" in str(e) and attempt < max_retries:
                    # Exponential backoff for resource unavailable errors
                    wait_time = (2 ** attempt) * 60  # 1, 2, 4 minutes
                    print(f"Flex processing resources unavailable. Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                elif e.status_code == 408 and attempt < max_retries:
                    # Handle timeout errors
                    wait_time = (2 ** attempt) * 30  # 30, 60, 120 seconds
                    print(f"Request timeout. Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
        
    def call_model(self, 
                   prompt: str, 
                   format_spec: any = None, 
                   tools: list = None,
                   parallel_tool_calls: bool = False,
                   override_service_tier: str = None):
        """
        Calls the OpenAI model using the /v1/responses API with new gpt-5 parameters.

        Args:
            prompt (str): The user prompt.
            format_spec (any, optional): Pydantic model for structured JSON output or a raw schema dict.
            tools (list, optional): A list of tool definitions, including custom tools with CFG.
            parallel_tool_calls (bool): Flag for parallel tool calling. Required to be False for custom tools.
            override_service_tier (str): Override the default service_tier for this specific call.

        Returns:
            The response object from the OpenAI API.
        """
        def make_api_call():
            params = {
                "model": self.model,
                "input": [{"role": "user", "content": prompt}],
            }
            
            # Add service tier
            service_tier = override_service_tier or self.service_tier
            if service_tier:
                params["service_tier"] = service_tier
            
            # Add new reasoning parameter if not default
            if self.reasoning_effort != "medium":
                params["reasoning"] = {"effort": self.reasoning_effort}

            # Build the 'text' parameter dictionary
            text_param = {}
            if format_spec and isinstance(format_spec, dict): # For raw schema in batch emulation
                text_param = format_spec
            if self.verbosity != "medium":
                text_param["verbosity"] = self.verbosity
            
            if text_param:
                params["text"] = text_param

            # Add tools if provided
            if tools:
                params["tools"] = tools
                # Per docs, custom tools require parallel_tool_calls to be False
                if any(t.get('type') == 'custom' for t in tools):
                    params["parallel_tool_calls"] = False
                else:
                    params["parallel_tool_calls"] = parallel_tool_calls
            
            # Use .parse() for Pydantic models for automatic validation
            if format_spec and isinstance(format_spec, type) and issubclass(format_spec, pydantic.BaseModel):
                params["text_format"] = format_spec
                response = self.client.responses.parse(**params)
            else: # Use .create() for free-form, verbosity-controlled, or raw schema calls
                response = self.client.responses.create(**params)

            # Update usage tracking
            if self.model not in self.usage:
                self.usage[self.model] = {'input': 0, 'cached_input': 0, 'output': 0}
            
            if response.usage:
                # Track cached input tokens separately
                cached_tokens = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0)
                
                # Regular input tokens = total input tokens - cached tokens
                regular_input_tokens = response.usage.input_tokens - cached_tokens
                
                self.usage[self.model]['input'] += regular_input_tokens
                self.usage[self.model]['cached_input'] += cached_tokens
                self.usage[self.model]['output'] += response.usage.output_tokens
            
            return response

        return self._handle_api_call_with_retry(make_api_call)
    
    def call_model_with_fallback(self, 
                                 prompt: str, 
                                 format_spec: any = None, 
                                 tools: list = None,
                                 parallel_tool_calls: bool = False):
        """
        Calls the model with flex processing first, then falls back to standard processing if resources unavailable.
        Only applicable if the instance is configured for flex processing.
        
        Args:
            prompt (str): The user prompt.
            format_spec (any, optional): Pydantic model for structured JSON output or a raw schema dict.
            tools (list, optional): A list of tool definitions, including custom tools with CFG.
            parallel_tool_calls (bool): Flag for parallel tool calling.

        Returns:
            The response object from the OpenAI API.
        """
        if self.service_tier != "flex":
            # If not using flex, just call normally
            return self.call_model(prompt, format_spec, tools, parallel_tool_calls)
        
        try:
            # Try with flex processing first
            return self.call_model(prompt, format_spec, tools, parallel_tool_calls)
        except openai.APIError as e:
            if e.status_code == 429 and "Resource Unavailable" in str(e):
                print("Flex processing resources unavailable. Falling back to standard processing...")
                # Fallback to standard processing
                return self.call_model(prompt, format_spec, tools, parallel_tool_calls, override_service_tier="auto")
            else:
                raise e
    
    def create_batch_request(self, 
                             prompt: str, 
                             format_spec: any, 
                             custom_id: str = None, 
                             tools: list = None,
                             parallel_tool_calls: bool = False,
                             override_service_tier: str = None) -> dict:
        """
        Create a single batch request entry for the /v1/responses endpoint.
        """
        if custom_id is None:
            custom_id = f"req-{uuid4()}"

        body = {
            "model": self.model,
            "input": [{"role": "user", "content": prompt}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"{format_spec.__name__.lower()}_response",
                    "schema": to_strict_json_schema(format_spec),
                    "strict": True
                }
            }
        }

        # Add service tier
        service_tier = override_service_tier or self.service_tier
        if service_tier:
            body["service_tier"] = service_tier

        if self.reasoning_effort != "medium":
            body["reasoning"] = {"effort": self.reasoning_effort}
        
        if self.verbosity != "medium":
            body["text"]["verbosity"] = self.verbosity

        if tools:
            body["tools"] = tools
            if any(t.get('type') == 'custom' for t in tools):
                body["parallel_tool_calls"] = False
            else:
                body["parallel_tool_calls"] = parallel_tool_calls

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body
        }
    
    def test_batch_request(self, jsonl_string: str) -> list[dict]:
        """
        Test a JSONL string before batch processing by calling the API directly.
        """
        jobs = [json.loads(json_string) for json_string in jsonl_string.split('\n') if json_string]

        def process_job(job):
            try:
                # The prompt is in `input` and the format is in `text` now
                # Extract service_tier if present for individual call
                override_service_tier = job['body'].get('service_tier')
                response = self.call_model(
                    prompt=job['body']['input'][0]['content'],
                    format_spec=job['body']['text'],
                    override_service_tier=override_service_tier
                )
            except openai.APIError as e:
                print(f"API Error processing job {job.get('custom_id')}: {e}")
                return None
            except Exception as e:
                print(f"General Error processing job {job.get('custom_id')}: {e}")
                return None

            return {
                'custom_id': job['custom_id'],
                'response': {'body': json.loads(response.model_dump_json())}
            }

        max_workers = 50
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_job, jobs), total=len(jobs), desc="Processing Test Batch"))
        
        return [r for r in results if r is not None]

    def validate_stage1_response(self, response_dict: dict, xml_content: str) -> tuple[bool, str, str]:
        """
        Validate a stage1 response by running the generated script and checking output.
        
        Args:
            response_dict: The response dictionary from stage1
            xml_content: The original XML content to process
            
        Returns:
            tuple: (is_valid, csv_output, error_message)
        """
        try:
            # Extract the response data
            response_text = response_dict['response']['body']['output'][1]['content'][0]['text']
            response_data = json.loads(response_text)
            pyscript = response_data['pyscript']
            
            # Run script in isolated namespace
            isolated_globals = {
                '__builtins__': __builtins__,
                'xml_content': xml_content,
                # Add common imports that the script might need
                're': __import__('re'),
                'csv': __import__('csv'),
                'StringIO': StringIO,
                'ET': __import__('xml.etree.ElementTree', fromlist=['ElementTree'])
            }
            exec(pyscript, isolated_globals)
            xml_to_csv = isolated_globals['xml_to_csv']
            csv_output = xml_to_csv(xml_content)
            
            # Validate CSV has >= 2 data rows (excluding header)
            lines = csv_output.strip().split('\n')
            if len(lines) < 3:  # header + at least 2 data rows
                return False, csv_output, f"Insufficient data rows: got {len(lines)-1}, need >= 2"
            
            # Validate CSV has more than 1 column
            if lines:
                header_columns = len(lines[0].split(','))
                if header_columns <= 1:
                    return False, csv_output, f"Insufficient columns: got {header_columns}, need > 1"
            
            return True, csv_output, ""
            
        except Exception as e:
            return False, "", str(e)

    def test_and_retry_stage1(self, input_dir: str, n_files: int = None, max_attempts: int = 3) -> tuple[list[dict], list[dict]]:
        """
        Test stage1 batch with validation and retry logic - fully parallel processing.
        
        Args:
            input_dir: Directory containing XML files
            n_files: Number of files to process (None for all)
            max_attempts: Maximum attempts per file
            
        Returns:
            tuple: (successful_responses, failed_responses)
        """
        from src.utils import read_xml
        
        files = glob.glob(input_dir + '*.xml')
        random.Random(42).shuffle(files)
        if n_files:
            files = files[:n_files]
        
        def process_file_with_retries(file_path):
            """Process a single file with retry logic"""
            xml_content = read_xml(file_path)
            last_error = ""
            
            for attempt in range(max_attempts):
                try:
                    # Prepare XML content (truncate if too long)
                    table_xml_str = xml_content
                    lines = table_xml_str.splitlines()
                    if len(lines) > 300:
                        table_xml_str = '\n'.join(lines[:200] + ['...'] + lines[-100:])
                    
                    # Create prompt with retry context if needed
                    base_prompt = create_xml_to_csv_script_prompt(table_xml_str)
                    if attempt > 0:
                        retry_prompt = f"""
{base_prompt}

IMPORTANT: This is attempt {attempt + 1} of {max_attempts}. Previous attempts failed validation.
Previous error: {last_error}

Requirements that must be met:
1. The generated Python script must execute without errors
2. The resulting CSV must have at least 2 data rows (excluding header)
3. Pay special attention to proper row detection and data extraction

Please ensure your script properly handles edge cases and produces valid output.
"""
                        prompt = retry_prompt
                    else:
                        prompt = base_prompt
                    
                    # Make the API call
                    response = self.call_model(prompt, format_spec=Script)
                    response_dict = {
                        'custom_id': file_path,
                        'response': {'body': json.loads(response.model_dump_json())},
                        'attempt': attempt + 1
                    }
                    
                    # Validate the response
                    is_valid, csv_output, error_msg = self.validate_stage1_response(response_dict, xml_content)
                    
                    if is_valid:
                        print(f"✓ {file_path} succeeded on attempt {attempt + 1}")
                        return {'success': True, 'response': response_dict}
                    else:
                        last_error = error_msg
                        print(f"✗ {file_path} failed attempt {attempt + 1}: {error_msg}")
                        if attempt == max_attempts - 1:
                            # Final attempt failed
                            response_dict['validation_error'] = error_msg
                            response_dict['csv_output'] = csv_output
                            return {'success': False, 'response': response_dict}
                        
                except Exception as e:
                    last_error = str(e)
                    print(f"✗ {file_path} error on attempt {attempt + 1}: {e}")
                    if attempt == max_attempts - 1:
                        return {'success': False, 'response': {
                            'custom_id': file_path,
                            'error': str(e),
                            'attempt': attempt + 1
                        }}
            
            # This shouldn't be reached, but just in case
            return {'success': False, 'response': {
                'custom_id': file_path,
                'error': 'Unknown error - max attempts exceeded',
                'attempt': max_attempts
            }}
        
        # Process files in parallel
        max_workers = 50
        print(f"Processing {len(files)} files with {max_workers} parallel workers...")
        
        successful_responses = []
        failed_responses = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_file_with_retries, files), 
                total=len(files), 
                desc="Processing files with retry"
            ))
        
        # Separate successful and failed responses
        for result in results:
            if result['success']:
                successful_responses.append(result['response'])
            else:
                failed_responses.append(result['response'])
        
        print(f"\nResults: {len(successful_responses)} successful, {len(failed_responses)} failed")
        return successful_responses, failed_responses

    def validate_and_retry_existing_responses(self, batch_responses: list[dict], max_attempts: int = 3) -> tuple[list[dict], list[dict]]:
        """
        Validate existing batch responses and retry failed ones.
        
        Args:
            batch_responses: List of existing batch response dictionaries
            max_attempts: Maximum total attempts per file (including original)
            
        Returns:
            tuple: (successful_responses, failed_responses)
        """
        from src.utils import read_xml
        
        successful_responses = []
        failed_responses = []
        
        for response_dict in tqdm(batch_responses, desc="Validating and retrying responses"):
            custom_id = response_dict['custom_id']
            
            try:
                xml_content = read_xml(custom_id)
                is_valid, csv_output, error_msg = self.validate_stage1_response(response_dict, xml_content)
                
                if is_valid:
                    print(f"✓ {custom_id} - original response is valid")
                    successful_responses.append(response_dict)
                    continue
                
                print(f"✗ {custom_id} - original response failed: {error_msg}")
                
                # Retry with enhanced prompts
                success = False
                for attempt in range(1, max_attempts):  # Start from 1 since original was attempt 0
                    try:
                        table_xml_str = xml_content
                        lines = table_xml_str.splitlines()
                        if len(lines) > 300:
                            table_xml_str = '\n'.join(lines[:200] + ['...'] + lines[-100:])
                        
                        retry_prompt = f"""
{create_xml_to_csv_script_prompt(table_xml_str)}

IMPORTANT: This is retry attempt {attempt + 1} of {max_attempts}. Previous attempts failed validation.
Previous error: {error_msg}

Requirements that must be met:
1. The generated Python script must execute without errors
2. The resulting CSV must have at least 2 data rows (excluding header)
3. Pay special attention to proper row detection and data extraction

Please ensure your script properly handles edge cases and produces valid output.
"""
                        
                        response = self.call_model(retry_prompt, format_spec=TableScript)
                        new_response_dict = {
                            'custom_id': custom_id,
                            'response': {'body': json.loads(response.model_dump_json())},
                            'retry_attempt': attempt + 1
                        }
                        
                        is_valid, csv_output, error_msg = self.validate_stage1_response(new_response_dict, xml_content)
                        
                        if is_valid:
                            print(f"✓ {custom_id} succeeded on retry attempt {attempt + 1}")
                            successful_responses.append(new_response_dict)
                            success = True
                            break
                        else:
                            print(f"✗ {custom_id} failed retry attempt {attempt + 1}: {error_msg}")
                    
                    except Exception as e:
                        print(f"✗ {custom_id} error on retry attempt {attempt + 1}: {e}")
                        error_msg = str(e)
                
                if not success:
                    response_dict['validation_error'] = error_msg
                    response_dict['final_attempt'] = max_attempts
                    failed_responses.append(response_dict)
                    print(f"✗ {custom_id} failed all {max_attempts} attempts")
                    
            except Exception as e:
                print(f"✗ {custom_id} error during validation: {e}")
                response_dict['validation_error'] = str(e)
                failed_responses.append(response_dict)
        
        print(f"\nValidation Results: {len(successful_responses)} successful, {len(failed_responses)} failed")
        return successful_responses, failed_responses

    def submit_batch(self, jsonl_string: str) -> str:
        batch_file = self.client.files.create(
            file=StringIO(jsonl_string).getvalue().encode('utf-8'),
            purpose="batch"
        )
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/responses",
            completion_window="24h"
        )
        return batch

    def wait_on_batch(self, batch_id: str, poll_interval: int = 30) -> dict:
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status == "completed":
                output = self.client.files.content(batch.output_file_id)
                return [json.loads(line) for line in output.text.strip().split("\n")]
            elif batch.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch failed with status: {batch.status}")
            print(f"Batch {batch_id} status: {batch.status}. Waiting...")
            time.sleep(poll_interval)

    def create_stage2_batch_str(self, stage1_batch_response: list) -> str:
        batch_lines = []
        conn = get_sql_conn()
        for batch_response in stage1_batch_response:
            prompt = create_sql_prompt(conn, batch_response)
            if not prompt: continue
            batch_request = self.create_batch_request(prompt, format_spec=SQL, custom_id=batch_response['custom_id'])
            batch_lines.append(json.dumps(batch_request))
        return '\n'.join(batch_lines)
    
    def create_stage3_batch_str(self, stage1_batch_response: list, stage2_batch_response: list) -> str:
        batch_lines = []

        for stage2_response in stage2_batch_response:
            conn = get_sql_conn()
            custom_id = stage2_response['custom_id']
            stage1_response = next((r for r in stage1_batch_response if r['custom_id'] == custom_id), None)
            if not stage1_response: continue

            stage2_content = json.loads(stage2_response['response']['body']['output'][1]['content'][0]['text'])

            if not stage2_content: continue

            sql_command = stage2_content.get('sql_command')
            csv_data = get_csv_from_response(stage1_response)
            
            if not sql_command or not csv_data: continue
            
            add_secondary_sql_table(conn, csv_data)
            
            try:
                conn.execute(sql_command)
            except Exception as e:
                print(f"Error executing SQL for {custom_id}: {e}")
                continue
            primary_table = conn.sql("SELECT * FROM primary_table").df()
            secondary_table = conn.sql("SELECT * FROM secondary_table").df()
            if not primary_table.empty:
                table_context = read_xml(stage1_response['custom_id'].replace('.xml', '_context.txt'))
                prompt = create_chemistry_prompt(context=table_context, table_head=secondary_table.iloc[0].to_dict())
                batch_request = self.create_batch_request(prompt, format_spec=RowChemistry, custom_id=custom_id)
                batch_lines.append(json.dumps(batch_request))
        print(f'Created {len(batch_lines)} chemistry prompts')
        return '\n'.join(batch_lines)
    
    def create_stage4_batch_str(self, custom_ids: str, n_files: int = None) -> str:
        batch_lines = []
        for table_file in custom_ids:
            if n_files is not None and len(batch_lines) >= n_files: break
            try:
                table_context_file = table_file.replace('.xml', '_context.txt')
                table = read_xml(table_file)
                table_head, table_context = table[0:100], read_xml(table_context_file)
                prompt = create_metadata_prompt(table_context, table_head)
                batch_request = self.create_batch_request(prompt, format_spec=Metadata, custom_id=table_file)
                batch_lines.append(json.dumps(batch_request))
            except Exception as e:
                print(f"Error processing {table_file}: {e}")
        return '\n'.join(batch_lines)

    def stack_csv(self, prompt: str) -> dict:
        response = self.call_model(prompt, format_spec=SQL)
        return response.output_parsed.model_dump()

    def calculate_cost(self, is_batch: bool = False) -> dict:
        """
        Calculate costs per model with optional batch pricing.
        Prices are in USD per 1 million tokens.
        Flex processing automatically gets batch pricing rates.
        """
        prices = {
            'gpt-5': {'input': 1.25, 'cached_input': 0.125, 'output': 10.00},
            'gpt-5-mini': {'input': 0.25, 'cached_input': 0.025, 'output': 2.00},
            'gpt-5-nano': {'input': 0.05, 'cached_input': 0.005, 'output': 0.40},
        }
        
        costs = {}
        total_cost = 0.0
        for model, usage in self.usage.items():
            model_key = model.split('-202')[0] # Normalize model names like 'gpt-4o-2024-05-13' to 'gpt-4o'
            
            if model_key in prices:
                model_prices = prices[model_key]
                cost = (
                    (usage['input'] * model_prices['input'] + 
                     usage['cached_input'] * model_prices['cached_input'] +
                     usage['output'] * model_prices['output']) / 1_000_000
                )
                # Apply discounts: Batch API jobs get 50% discount, Flex processing gets batch rates
                is_discounted = is_batch or self.service_tier == "flex"
                final_cost = cost * 0.5 if is_discounted else cost
                costs[model] = final_cost
                total_cost += final_cost
            else:
                print(f"Warning: Price not found for model '{model_key}'")
                costs[model] = 0.0
            
        costs['total'] = total_cost
        return costs