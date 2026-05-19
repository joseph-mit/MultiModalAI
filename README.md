# MultiModal AI
Joseph Firmansyah, Stephanie Chen, Sophie Lin
## About
Repository for Multimodal AI class project.
This research project uses multimodal learning to analyze real estate listings by combining Zillow listing data, Google Street View imagery, satellite images, and census data, to study their interactions and to draw useful inferences in novel ways.
## Repository Structure
- `proposal/` - project proposal
  - `Multimodal_AI_Project_Proposal.pdf` - initial project proposal
  - `PROPOSAL_README.md`
- `midterm/` - files containing midterm report
  - `data_pipeline/` - Python scripts for data collection, cleaning, and maintenance
    - `data_collection.py` - Scrapes Zillow listings, GSV, and satellite images
    - `data_cleaning.py` - Remaps paths and computes price gaps
    - `data_maintenance.py` - Fills missing census data, images, and listing photos
    - `clip_script.py` - Encodes listing text and photos using CLIP embeddings
  - `figures/` - Data visualizations and result plots
  - `MidtermPaper.ipynb` - Python code for studies, experiements, and data visualizations for midterm project report
  - `Multimodal_AI_Project_Midterm.html` - Midterm project report
  - `MIDTERM_README.md`
- `final/` - files containing final report
  - `data_pipeline/` - Python scripts for data collection, cleaning, and maintenance
