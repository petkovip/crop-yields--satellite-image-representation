- important to see context + development over time!
- Random observations:
	- manmade things are straight lines - e.g. deforestation
	- military assembles/moves stuff in the same way before invading and generally do things in repetative ways
- Data is RGB + NIR spectral
- **NDVI**: live green plants re-emit solar radiation in the near-infrared (**NIR**) spectral region, but they absorb solar radiation in the photosynthetically active radiation (**PAR**) - Chlorophil (the pigment in plant leaves) strongly absorbs visible light
	- so they are bright in NIR and dark in PAR
	- --> ***NDVI = (NIR - Red) / (NIR + Red)***, where Red is the reflectance measurement in visible region, while NIR - in the near infrared region
- but NDVI (similar to bag of words) fails to capture context in the imagery as it discards structure
- **Handcrafted features** - statistical textures of a region, e.g. how bumpy or uniform a field looks
- proposed **Geo-SimCLR** (positive temporal pairs) is ~ to BERT (masked words and next sentence prediction) conceptually
- Fine-tuning - can later train Geo-SimCLR to predict commodity returns (e.g. with LSTM)
- Need to carefully select **positive and negative pairs** to account for crop calendar (e.g. comparing Jan to July is meaningless, but comparing June to July is likely meaningful) as well as acro-climatic zones (e.g. comparing fields near each other is a positive pair, while fields in different zones are negative pairs)
- ***Better to predict USDA yields or condition reports rather than CME futures returns*** (as the latter is noisier - e.g. weather forecast is priced in markets before crops show signs of the realized weather); then could investigate potentially how USDA yields predict financial asset returns at a later point
- **Data**:
	- used CropNet
		- initial subset has 3 states x ~98 counties x ~12 bi-weeks x 5 years
		- mean of 12 grids per county but varies
		- AG dataset - state, year, quarter, 
	- initially do the full data loading and NN training on a single county
	- then generalize to 1 state (note 1 state is fine even for performance evaluation since USDA publishes ***weekly "Crop Progress and Condition"*** reports - rating Very Poor, Poor, Fair, Good, Excellent - ***at state level***; in addition, National Agricultural Statistics Service publishes ***annual crop yields per county***, e.g. 102 distinct values just in the state of Illinois)
	- download or transfer to google drive the processed image datasets to Yale HPC directory
	- write a *?SLURM?* script to submit the job to an A100 GPU node
- important to double check the code in Cursor correctly does the custom contrastive loss functions and generally our *?ResNet?* model

# Project and questions:
- negative samples - far in time, different states, and different CRD (Crop Reporting District) within a state
- positive samples - close in time, same CRD in same state
- Takeaways - call with Jake:
	- potentially use coordinates for negative/positive sampling
	- do a 3rd OOS test - across crops
	- PHATE etc. is important to visiualize if things look ok when training even if i fail to beat the established models
- Geo-SimCLR trained - LR seems low for me, but if performance not good enough after comparison, can retrain with:
	- LR = 1e-3                # 3x higher (proportional to sqrt(2) for batch 512)
	- WARMUP_EPOCHS = 3        # linear warmup from LR/100 -> LR over first 3 epochs  
	- EPOCHS = 50              # extend training, no harm if loss already plateaued
	- BATCH_SIZE = 512         # with MULTI_GPU=True
	- MULTI_GPU = True

# Papers:

### Deep Learning for Agriculture Satellite Imagery papers/datasets:
- [Dataset](https://huggingface.co/datasets/CropNet/CropNet) and [paper](https://dl.acm.org/doi/10.1145/3637528.3671536) - highly relevant dataset
- Prithvi-EO-2.0 model - [hugging face](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
All of the papers/datasets below are from the ["Datasets for deep learning applied to satellite and aerial imagery" repo](https://github.com/satellite-image-deep-learning/datasets) (or related to papers in that set) which links many satellite imagery sources/papers/datasets/competitions:
- [A CNN-RNN Framework for Crop Yield Prediction](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2019.01750/full) - HIGHLY RELEVANT
	- crop yield prediction of corn and soybeans across the entire Corn Belt in the US for 2016-18
	- [repo is here](https://github.com/saeedkhaki92/CNN-RNN-Yield-Prediction)
	- the csv dataset (which i am guessing is just yield data) is saved in readings\CNN-RNN Framework for Crop Yield Prediction
	- can email saeed_khaki@outlook.com to request the (300GB) dataset
- [SpatioTemporalYield](https://huggingface.co/datasets/ellaampy/SpatioTemporalYield)
- some random paper which might have data - [Using PySpark for Image Classification on Satellite Imagery of Agricultural Terrains](https://github.com/hellosaumil/deepsat-aws-emr-pyspark)
- [Space2Ground](https://github.com/Agri-Hub/Space2Ground) - dataset with Space (Sentinel-1/2) and Ground (street-level images) components, annotated with crop-type labels for agriculture monitoring
- [Global Fields of The World (FTW)](https://source.coop/ftw/global-data) - global-scale estimates of agricultural fields for 2024–2025. The dataset includes both model inputs (Sentinel-2–derived median composites in COG and Zarr v3 formats) and outputs (in Zarr, GeoParquet and PMTiles)
- AI4Boundaries - open AI-ready dataset to map field boundaries with Sentinel-2 and aerial photography
	- [repo](https://github.com/waldnerf/ai4boundaries)
	- [paper](https://essd.copernicus.org/articles/15/317/2023/essd-15-317-2023-discussion.html)
- there are some datasets specific for a region in italy, france and india i found, but probs not that relevant for this project
- Enhanced Sentinel 2 Agriculture Challange (already closed) with a starter pack [repo](https://github.com/AI4EO/enhanced-sentinel2-agriculture-challenge?ref=philabchallenges-cms.earthpulse.es) but i think it focuses on slovenia (at least the challange does, not sure about whether one can get broader data but in any case it is likely to be european)
- [Sen4AgriNet](https://github.com/Orion-AI-Lab/S4A) - A Sentinel-2 multi-year, multi-country benchmark dataset for crop classification and segmentation with deep learning, with and [models](https://github.com/Orion-AI-Lab/S4A-Models)
- [AgriPotential](https://github.com/MohammadElSakka/agripotential) - a Satellite Image Time Series (STIS) of 34 Sentinel-2 timeframes with 5 classes of agriculutural potential
	- agripotential tutorial colab notebook is [here](https://colab.research.google.com/drive/1Ys_z7qBY1pa8iY7PMaLzDsTuiFaJDhHu#scrollTo=7YdCq3SOBFcc)
	- related to a competition ["AgriPotential: Viticulture Edition"](https://www.codabench.org/competitions/12055/) on viticulture prediction; this competition is the only one that's happening right now, but the registration closed 4 days ago. If relevant I could email registration@clef-initiative.eu and request to be registered late (not sure if it will work)
- PAPER: [Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks](https://arxiv.org/abs/2107.07933)
- PAPER: [Agriculture-Vision: A Large Aerial Image Database for Agricultural Pattern Analysis](https://arxiv.org/abs/2001.01306)
	- [repo](https://github.com/dapsavoie/agricultural_satellite_classifier?utm_source=catalyzex.com)
	- latest [Agriculture-Vision page](https://www.agriculture-vision.com/)
	- [Agriculture-vision ?old? database](https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021)
- Paper: [OpenSatMap: A Fine-grained High-resolution Satellite Dataset for Large-scale Map Construction](https://arxiv.org/html/2410.23278v1) - broader

### Machine-Learning:
- **[SimCLR]([A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709))**; Article - [Advancing Self-Supervised and Semi-Supervised Learning with SimCLR (Google Research, 2020)]([Advancing Self-Supervised and Semi-Supervised Learning with SimCLR](https://research.google/blog/advancing-self-supervised-and-semi-supervised-learning-with-simclr/))
	- pull similar things together (positive pairs), push different things apart (negative pairs)
	- Process:
		- 1. creates positive pairs via augmentations - random cropping, color distortion, rotation, Gaussian blur, etc.
		- 2. computes image representation via CNN based on [ResNet](https://arxiv.org/pdf/1512.03385) architecture
		- 3. computes non-linear image representations using MLP, which amplifies the invariant features
		- 4. uses SGD to update both the CNN and MLP
		- 5. either use representations directly or fine-tune with labeled images for downstream tasks
	- finds that from the augmentations random cropping and random color distortion stand out (although neither leads to high performance on its own, their combo does); the nonlinear projection is important, but surprising results for the linear representation (i.e. end of CNN before MLP starts); also, scaling up examples within a batch, width and depth, and more epochs all lead to improvement in performance
	- [Code]([GitHub - google-research/simclr: SimCLRv2 - Big Self-Supervised Models are Strong Semi-Supervised Learners · GitHub](https://github.com/google-research/simclr))
- NT-Xent - [explanation and code]([NT-Xent (Normalized Temperature-Scaled Cross-Entropy) Loss Explained and Implemented in PyTorch | Towards Data Science](https://towardsdatascience.com/nt-xent-normalized-temperature-scaled-cross-entropy-loss-explained-and-implemented-in-pytorch-cc081f69848/)):
	- normalized with cosine similarity between 2 pictures; 
	- temperature scaled - low temp amplifies highest prob, reduces low prob
	- then compute cross entropy loss
	- NT-BXent (normalized temperature-scaled binary cross entropy loss) - same as NT-Xent but uses binary cross entropy for more than 2 augmentations (i.e. ~ multi-label)
- DINO:
	- [v1-paper](https://arxiv.org/pdf/2104.14294)
	- [v2-article](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/)
	- [v3-article](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)
- SigLIP
	- [v1-paper](https://arxiv.org/pdf/2303.15343)
	- [v2-paper](https://arxiv.org/pdf/2502.14786?)
- PHATE
	- [M-PHATE paper](https://arxiv.org/pdf/1908.02831)
	- 
- [t-SNE, UMAP]([pdf](https://openreview.net/pdf?id=B8a1FcY0vi)) and [On UMAP's True Loss Function]([On UMAP's True Loss Function](https://proceedings.neurips.cc/paper/2021/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html)) 

### Finance-focused ones:
- [Empirical Asset Pricing via Machine Learning (Gu, Kelly, Xiu - 2020)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577)
	- Applies different ML methods to predict asset risk premia and compares their performance; argues ML methods capture non-linear interactions between predictors
	- useful for the financial tests to determine if my predictor is good

### Big Data Impact in Finance:
- **[Institutional trading and satellite data (Ha, 2025)](https://www.sciencedirect.com/science/article/pii/S1544612324013709):**
	- access to satellite imagery enhances return predictability of daily institutional trading - more pronounced for stocks with severe info asymmetry and is driven by non-HFs
	- implies institutions, mainly non-HFs, actively adjust positions daily based on satellite data
	- RS Metrics for satellite data and ANcerno for institutional clients' transaction records
***TO READ LITERATURE REVIEW AND THEN UNDERSTAND DATA SOURCES AND DATA STRUCTURE, THEN OBTAIN AN PLAY WITH DATA***
- **Big Data as a Governance Mechanism (Zhu, 2019):**
	- initial study of the impact of big data (consumer transactions and satellite imagery) on investment efficiency and price informativeness
- **[Eye in the sky: Private satellites and government macro data (Mukherjee et.al., 2021)](https://www.sciencedirect.com/science/article/pii/S0304405X21000921):**
	- studies whether satellite-based macro estimates can replace government-issued macro data by using cloudier hubs as control group
	- finds high usage of satellite data to the point that macro announcements are no longer surprises to markets
	- thus macro uncertainty resolution is smoother and governments have less control over info
- **Displaced by Big Data? Evidence from Active Fund Managers (Bonelli and Faucault, 2024):**
	- studies how managers without access to satellite data reacted to other investors using it
	- finds they either divest from these stocks (retailers based on parking lot data) or they have another edge not captured by this big data
- **On the Capital Market Consequences of Big Data: Evidence from Outer Space (Katona et al., 2024):**
	- finds that sophisticated investors with access to satellite data can formulate profitable trading strategies, e.g. targeting upcoming earnings reports of retailers by tracking parking lots activity
	- this led to more informed short selling activity and less informed activity around the actual earnings
	- unequal access to this data increases asimmetry among market participants, but does not enhance price discovery that much (perhaps since access to this data and its usage are costly, so investors need to be compensated for accessing it)
	- uses data from RS Metrics and Orbital Insight

# People:
- [Gjorgjina Cenikj](https://www.linkedin.com/in/gjorgjina-cenikj-310693167/)
	- ML engineer / researcher at Planet


# Ideas:
- colour of fields (brown vs green)
- water / draught of nearby water pools