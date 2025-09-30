import duckdb
import json
from src.sql import add_secondary_sql_table, get_sql_types, get_sql_head
from src.utils import get_csv_from_response

def create_xml_to_csv_script_prompt(table_xml_str: str) -> str:
    return f"""
    ## Task
    Write a Python 3.11 script to convert OCR-extracted table XML data into a structured CSV format.
    
    ## Output Format
    Return a Script object with:
    - `pyscript`: Complete Python conversion script as a string
    
    ## Function Requirements
    - The Python script should contain a function `xml_to_csv(xml_str: str) -> str`
    - You do need to return if __name__ ... w/ example usage - just our function.
    
    ## Technical Requirements
    ### Dependencies
    - Uses only Python standard library + `re` module
    
    ### Column Name Rules
    - Preserve meaning of orignal column headers in new csv. i.e "UTC Untreated control group (%)" → "utc_untreated_control_group_pct" / "Inhibition (%)" → "inhibition_pct"
    - Make SQL-compatible: underscores for spaces, no dots, lowercase
    - Hardcode column names (no need to dynamically generate)
    
    ### Data Handling
    - Use "NA" for missing/empty cells
    - Properly escape CSV values (quotes, commas, newlines)
    - Use comma delimiter
    - Sometimes &#x2003; is used in the XML, this should be replaced with a space in the CSV.
    
    ## Domain Context
    - UTC = "Untreated Control" percentage - this is not inhibition
    - Preserve scientific notation and decimal precision
    - Don't interpret abbreviations unless obvious

    ## Quality control
    - To ensure correct rows let's strip newspace and capitalise the sequence column. It should have >=8 ATGC characters. If not skip the row.
    
    ## Preview of the input XML Structure (xml_str) - we will use the full version as input for your script (do not return this in your output!):
    {table_xml_str}
    """

def create_sql_prompt(conn: duckdb.duckdb.DuckDBPyConnection, response: dict) -> str:
    
    # Handle script-based response instead of direct CSV
    try:        
        csv_output = get_csv_from_response(response)
        
    except Exception as e:
        print(f"Error executing script: {e}")
        return False
    
    if not add_secondary_sql_table(conn, csv_output):
        # CSV not loaded:
        return False

    table_types = get_sql_types(conn)
    table_head = get_sql_head(conn)

    if table_types == None or table_head == None:
        return False

    return f"""
    Your goal is to produce a dataset of antisense-oligonucleotide sequences and their inhibition percentages.
    To do so you must stack the secondary_table with the primary_table using a SQL command.

    Required Columns in secondary_table:
    1. ASO sequence (case insensitive)
    2. One of:
       - inhibition/knockdown/reduction percentage
       -  UTC (Untreated Control) / RNA percentage
    
    Transformation Rules:
    - Numeric columns -> DOUBLE
    - inhibition_percent = 
        - Direct copy from inhibition/knockdown columns
        - 100 - UTC(%) for untreated control
    - CONCAT two columns if ASO sequence is split (e.g.,sequence_part_one, sequence_part_two)
    - When using CAST be careful, some rows may not be castable to double, hence use TRY_CAST.
        
    Task:    
    1. Generate SQL:
       - Stack (INSERT INTO) secondary_table onto primary_table
       - Apply transformations as needed
       - **CRITICAL: Always use "secondary_table" as the table name in your FROM clause**
       - **When referencing columns from secondary_table, use the exact column names shown in the schema, including any special characters or numbers.**

    Output Format:
    - sql_command: string containing complete SQL command to stack the secondary_table onto primary_table

    Data:
    primary_table:
    Schema:
    - aso_sequence_5_to_3 (VARCHAR): 5'-3' ASO nucleotide sequence
    - inhibition_percent (DOUBLE): target inhibition percentage, range 0-100

    secondary_table:
    Types: {table_types}
    First 5 rows: {table_head}
    """

def create_chemistry_prompt(context: str, table_head: str) -> str:
    return f"""
    Task:
    You must create a python function to map a table row containing an antisense oligonucleotide and return formatted chemistry. You may use the table context and row description to aid your response.

    The python function must return a Chemistry class:

    class Modification(BaseModel):
        modification: str = Field(
            ...,
            title="Modification Name",
            description="Name of the modification. Sugar modifications: MOE (2'-O-methoxyethyl), cEt ((S)-constrained ethyl), LNA (locked nucleic acid), OMe (2'-O-methyl). Backbone modification: PS (phosphorothioate). Default sugar is D (deoxyribonucleic), default backbone is PO (phosphodiester)"
        )
        type: Literal["sugar", "backbone"] = Field(
            ...,
            title="Modification Type",
            description="Type of the modification - either 'sugar' (modifies nucleotide sugar ring) or 'backbone' (modifies internucleotide linkage)"
        )
        positions: List[int] = Field(
            ...,
            title="Modification Positions",
            description="List of 1-based positions where this modification occurs. For sugar modifications, position N refers to nucleotide N. For backbone modifications, position N refers to the linkage between nucleotides N and N+1. Default chemistry (D/PO) is used for unspecified positions"
        )
        
    class Chemistry(BaseModel):
        length: int = Field(
            ...,
            title="Sequence Length",
            description="Total length of the oligonucleotide sequence. Each position has a sugar modification (default: D) and positions 1 to length-1 have a backbone linkage (default: PO)"
        )
        modifications: List[Modification] = Field(
            ...,
            title="Modifications",
            description="List of all modifications applied to the sequence. Each modification specifies type (sugar/backbone), positions, and chemistry. Sugar options: MOE, cEt, LNA, OMe (default: D). Backbone options: PS (default: PO)"
        )

    Output format:
        - process_row_py: Executable Python function (def process_row_chemistry(row: dict) -> Chemistry) that processes a row of the table and returns a valid Chemistry class. Modification and Chemistry classes are already defined in the environment as above. DO NOT return anything else after the function (testing etc.)

    EXAMPLE 1:
    INPUT:
      Table Context: " ... are 16 nucleotides in length wherein ... The ‘Chemistry’ column describes the sugar modifications of each oligonucleotide. ‘k’ indicates an cEt sugar modification; ‘d’ indicates deoxyribose; and ‘e’ indicates a MOE modification..."
      Example Row: {{"sequence":"AAACCCAAATCCTCAT",...,"motif":"ekkeekkdddddddkk"}}
    OUTPUT:
    def process_row_chemistry(row: dict) -> Chemistry:
    # Get sequence length and motif
    sequence = row["sequence"]
    motif = row["motif"]
    length = len(sequence)
    
    # Parse MOE positions
    moe_positions = [i+1 for i, c in enumerate(motif) if c == 'e']
    
    # Parse cEt positions  
    cet_positions = [i+1 for i, c in enumerate(motif) if c == 'k']
    
    # Create Chemistry instance
    return Chemistry(
        length=length,
        modifications=[
            Modification(
                modification="MOE",
                type="sugar",
                positions=moe_positions
            ),
            Modification(
                modification="cEt", 
                type="sugar",
                positions=cet_positions
            ),
            Modification(
                modification="PS",
                type="backbone",
                positions=list(range(1, length))  # PS throughout
            )
        ]
    )
 
    Input data:
    Table Context: {context}
    Example Row: {table_head}
    """

def create_metadata_prompt(context: str, table_head: str) -> str:
    return f"""
    Task:
    Extract the name of the target gene, cell-line and dosage of the antisense oligonucleotide qRT-PCR experiment from the table context.
    If data is not present, return NA for string fields and -1 for numeric fields.

    Output format:
    - target_mrna (string): Return only the mRNA target without additional terms like "mRNA". If the target is reported as "HBV mRNA", return just "HBV".
    - target_gene (string): Return only the HGNC gene symbol corresponding to the target_mrna.
    - cell_line (string): e.g. HCT116
    - cell_line_species (string): Must be one of: "human", "mouse", "other"
    - dosage (float): Return only the numeric value in nM. For example, if "100nM" is reported, return 100.0
    - cells_per_well (int): Return only the numeric value. For example, if "100 cells per well" is reported, return 100.
    - transfection_method (string): Return only the method used to transfect the cells. Either ["Electroporation", "Gymnosis", "Lipofection"]

    Table Context: {context}
    Table Head: {table_head}
    """