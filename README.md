# Integrating higher-level semantics into robust biomedical name representations

This directory contains source code for the following paper:

`Integrating Higher-Level Semantics into Robust Biomedical Name Representations.` \
Pieter Fivez, Simon Šuster and Walter Daelemans. *LOUHI (EACL)*, 2021.

If you use this code, please cite:
    
    @inproceedings{fivez-etal-2021-integrating,
    title = "Integrating Higher-Level Semantics into Robust Biomedical Name Representations",
    author = "Fivez, Pieter  and
      Suster, Simon  and
      Daelemans, Walter",
    booktitle = "Proceedings of the 12th International Workshop on Health Text Mining and Information Analysis",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "49--58"}

## License

GPL-3.0

## Requirements

All requirements are listed in **requirements.txt**. 

You can run `pip install -r requirements.txt`, preferably in a virtual environment.

The fastText model used in the paper can be downloaded from the following link: 
https://drive.google.com/file/d/1B07lc3eeW_zughHguugLBR4iJYQj_Wxz/view?usp=sharing \
Our example script requires a path to this downloaded model.

## Data

Since we are not allowed to share SNOMED-CT data, we demonstrate our code using the openly available MedMentions corpus. \
We have used this corpus as fine-grained synonym sets in this previous publication:

    @inproceedings{fivez-etal-2021-conceptual,
    title = "Conceptual Grounding Constraints for Truly Robust Biomedical Name Representations",
    author = "Fivez, Pieter  and
      Suster, Simon  and
      Daelemans, Walter",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "2440--2450"}

The source files for this corpus can be found at https://github.com/chanzuckerberg/MedMentions.

The script **data/extract_medmentions.py** has used these source files to create **data/medmentions.json**.

## Code

We provide a script to run our training objectives from the paper. 

**main_dan.py** trains and evaluates the DAN encoder on **data/medmentions.json**. \
Please run `python main_dan.py --help` to see the options, or check the script. \
The default parameters are the best parameters reported in our paper.
