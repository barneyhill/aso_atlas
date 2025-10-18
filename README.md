# ASO Atlas

A comprehensive dataset of 188,521 RNase H-mediated antisense oligonucleotides (ASOs) with experimentally validated efficacy measurements, extracted from 417 USPTO patents published between 2001 and 2025.

<img width="7200" height="7200" alt="gene_circle" src="https://github.com/user-attachments/assets/4f996569-f2a1-43f0-9fcc-6febc383536c" />

## Overview

ASO Atlas provides the first large-scale, publicly available resource for understanding the relationship between ASO sequence, chemical modifications, and inhibitory activity. The dataset includes:

- **188,521 gapmer ASOs** targeting 306 human genes
- **Sequence composition** and target genomic locations
- **Chemical modification patterns** (phosphorothioate backbones, 2'-MOE, cEt sugar modifications)
- **Quantitative efficacy measurements** from qRT-PCR experiments across multiple cell lines
- **Experimental metadata** including cell type, dosage, and transfection method

## Installation

```bash
git clone https://github.com/barneyhill/aso_atlas.git
cd aso_atlas
pip install -r requirements.txt
```

## Quick Start

```python
import pandas as pd

# Load the dataset
df = pd.read_pickle('data/aso_atlas.pkl')

# View schema
print(df.columns)
# ['aso_sequence_5_to_3', 'inhibition_percent', 'chemistry', 'custom_id', 
#  'target_mrna', 'target_gene', 'cell_line', 'dosage', 
#  'cells_per_well', 'transfection_method', ...]
```

## Related Resources

- **OligoAI Model**: [https://huggingface.co/barneyhill/OligoAI](https://huggingface.co/barneyhill/OligoAI)
- **OligoAI Training Code**: [https://github.com/barneyhill/OligoAI](https://github.com/barneyhill/OligoAI)
- **Web Interface**: [https://sitlabs.org/OligoAI](https://sitlabs.org/OligoAI)
- **Preprint manuscript** [https://github.com/barneyhill/aso_atlas/blob/main/oligoai_manuscript.pdf]
