# MultiModal AI
Joseph Firmansyah, Stephanie Chen, Sophie Lin
## About
Course repository for Multimodal AI. This project uses multimodal learning to analyze real estate listings by combining Zillow listing data, Google Street View imagery, and satellite images with CLIP embeddings.
## Repository Structure
- `proposal` - project proposal
  - `Multimodal_AI_Project_Proposal.pdf` - final proposal submission
  - `PROPOSAL_README.md`
-`midterm/` - files containing midterm report 
  - `data_pipeline/` - Python scripts for data collection, cleaning, and maintenance
    - `data_collection.py` - Scrapes Zillow listings, GSV, and satellite images
    - `data_cleaning.py` - Remaps paths and computes price gaps
    - `data_maintenance.py` - Fills missing census data, images, and listing photos
    - `clip_script.py` - Encodes listing text and photos using CLIP embeddings
  - `figures/` - Visualizations and result plots
  - `MidtermPaper.ipynb` - Midterm paper
  - `Multimodal_AI_Project_Midterm.html` - Midterm project report
  - `MIDTERM_README.md`
