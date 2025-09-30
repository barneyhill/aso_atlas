import gzip
import os
import json

def read_xml(xml_file: str) -> str:
    if xml_file.endswith('.gz'):
        with gzip.open(xml_file, 'rt', encoding='utf-8') as f:
            return f.read().strip('\t')
    else:
        with open(xml_file, 'r', encoding='utf-8') as f:
            return f.read().strip('\t')
        
from io import StringIO
from src.utils import read_xml

def get_csv_from_response(response: dict) -> str:
    try:
        # Use the correct path from your actual data structure
        response_text = response['response']['body']['output'][1]['content'][0]['text']
        response_content = json.loads(response_text)
    except:
        return False
    
    # Handle script-based response instead of direct CSV
    try:
        custom_id = response.get('custom_id')  # Assuming custom_id is available in response

        # Get the original XML content
        xml_content = read_xml(custom_id)


        pyscript = response_content['pyscript']
        
        # Execute the script to get CSV output
        isolated_globals = {
            '__builtins__': __builtins__,
            're': __import__('re'),
            'csv': __import__('csv'),
            'StringIO': StringIO,
            'ET': __import__('xml.etree.ElementTree', fromlist=['ElementTree'])
        }
        exec(pyscript, isolated_globals)
        xml_to_csv = isolated_globals['xml_to_csv']
        csv_output = xml_to_csv(xml_content)
        
    except Exception as e:
        print(f"Error executing script: {e}")
        return False
    
    return csv_output