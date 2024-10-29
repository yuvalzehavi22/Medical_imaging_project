### Project overview:

This repository provides the code for our project, which enhances the HetMed model (https://arxiv.org/abs/2211.15158) by integrating advanced biomedical models, applying it to a new dataset, and utilizing multi-modal data for comprehensive patient analysis, with a focus on improving patient classification accuracy, particularly for glaucoma diagnosis.

The HetMed model is extended to incorporate medical images, clinical notes, and demographic details. Fine-tuned BioMedCLIP and ResNet-50 embeddings capture detailed patient image data, which are then combined with text-based and demographic features via the DMGI model to create a unified multi-modal representation. Each phase of the model was trained separately, with hyperparameter optimization performed using Optuna to achieve optimal performance.

This repository includes scripts for data preprocessing, image embedding extraction, clinical text feature extraction, model training, and evaluation.

### FairVLMed Dataset Overview:
This dataset focuses on Glaucoma and contains information on 10,000 patients. The dataset includes the following features:
Demographic Attributes: Contains demographic information such as age, gender, race, ethnicity, language, and marital status.
Labels: Indicates whether each patient has been diagnosed with Glaucoma.
Clinical Notes: According to the dataset's original publication, the clinical notes include not only descriptions of the images but also additional medical information, such as medications, lab tests, family medical history, etc.
Fundus Images: Images of the retina (fundus images) for each patient, providing visual data critical for Glaucoma diagnosis.

### How to download (FairVLMed  dataset)
- Download FairVLMed  dataset from https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP/tree/main .
- Save FairVLMed Fundus images in format .npy, and data_summary.csv.
- Save images in separate folders inside "dataset" folder: test_extracted_images, training_extracted_images, validation_extracted_images

### How to Run
- After saving images, to get image representation in SimCLR folder run
<pre><code>
{python run_model.py}
</code></pre>
- After get image representation and non-image feature, then run 
<pre><code>
{python kmeans_with_text.py} # if you want to construct graph based on features extracted from clinical notes.
or
{python kmeans.py} # if you want to construct graph based on demographic features.
</code></pre>

- After getting the pkl containing graphs, run DMGI model in MultiplexNetwork/main.py
<pre><code>
{python main.py}
</code></pre>
