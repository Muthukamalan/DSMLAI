# Early detection of lung cancer
general idea is that this is *what you should be prepared to do* for your project to succeed


- [X] S9  Setup ENV
- [ ] S10 Prepare PyTorch Dataset
    - data-parsing
    - data-manipulation
- [ ] S11 Intro classification Model
    - data formats
    - data sources
    - exploring the constraints that our problem domain places on us.
- [ ] S12 Model: metrics and how well the dataset is training and implement Solution
- [ ] S13 Creating a segmentation model that produces a heatmap helps to generate locations to classify
- [ ] S14 combine our segmentation + classification models to perform a final diagnosis
    



`Note`:

CT scans are essentially 3D X-rays, represented as a 3D array of single channel data. 
- Input: 3D CT scans of human torsos
- Output: location of suspected malignant tumors if any exists


Each voxel (volumetric data) of a CT scan has a numeric value that roughly corresponds to the aver age mass density of the matter contained inside. Most visualizations of that data show high-density material like bones and metal implants as white, low-density air and lung tissue as black, and fat and tissue as various shades of gray. 

<div align='center'>
<img src="./data/fight cancer using pytorch.png" width=1000 height=600/>
</div>


- Load our raw CT scan data into a pytorch tensor
- Identify the voxel of potential tumors in the lungs using Pytorch to implement a technique known as segmentation.
- Grouping interesting voxels into lumps called *nodules*. Each  nodule can be located by index,row and column of its center point.
- classify candidate noduels as actual nodules or non-nodules using 3D Conv
- Diagnose the patient using the combined per-nodule classifications.