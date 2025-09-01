#  ContraVirt: Wind Nowcasting in Unobserved Regions

##  Introduction
**ContraVirt** is a framework for **wind forecasting in regions without meteorological stations**.  
It integrates **real stations** with **virtual nodes** (representing unobserved locations) and combines **graph diffusion**, **contrastive learning**, and **multi-step forecasting**.

---

##  Installation
```bash
git clone https://github.com/JieJieNiu/-Wind-Nowcasting-in-Unobserved-Regions.git
pip install -r requirements.txt
```

---



## 📂 Project Structure
```
.
├── data/                  # KNMI dataset + processed graphs
│   ├── raw/               # Original CSV files
│   ├── processed/         # Graph tensors (graph_seq.pt, nodes_df.csv)
│   └── station_info.csv   # Metadata of weather stations
├── models/                # Model architectures
│   ├── temporal_gcn.py
│   ├── moco_module.py
│   └── ...
├── train.py          # Training script
├── test.py                # test script
├── args.py                
├── evaluation.py       # Plotting scripts (seasonal maps, embeddings)
└── README.md              # Project documentation

---
## Usage
### Train model
```bash
python train.py --args.MODEL_NAME Multi_step_MoCo --args.enable_multi_step_moco = True --epochs 200 --batch_size 32
python train.py --args.MODEL_NAME Augmented_MoCo --args.enable_augmented_moco = True --epochs 200 --batch_size 32
python train.py --args.MODEL_NAME Multi_step --args.enable_multi_step= True --epochs 200 --batch_size 32
python train.py --args.MODEL_NAME Augmented --args.enable_augmented = True --epochs 200 --batch_size 32
python train.py --args.MODEL_NAME w/o contrastive --args.enable_contrastive = False --epochs 200 --batch_size 32
python train.py --args.MODEL_NAME w/o contrastive&diffusion --args.DIFFUSION_METHOD="raw" --args.enable_contrastive = False --epochs 200 --batch_size 32

```

### Test
```bash
python test.py --args.MODEL_NAME Multi_step_MoCo 
python test.py --args.MODEL_NAME Augmented_MoCo
python test.py --args.MODEL_NAME Multi_step 
python test.py --args.MODEL_NAME Augmented
python test.py --args.MODEL_NAME w/o contrastive
python test.py --args.MODEL_NAME w/o contrastive&diffusion
```
---

### Evaluation



## 🏃 Usage

### 1. Preprocess dataset
```bash
python create_dataset.py --data_dir ./data
```

### 2. Train a model
```bash
python train_loop.py --model Multi_step_MoCo --epochs 200 --batch_size 32
```

### 3. Evaluate on test stations
```bash
python test.py --model Multi_step_MoCo --output_dir ./results
```

---

## 📊 Evaluation Protocol
- **Stations**: 8 held-out stations (not used in training)  
- **Metrics**:  
  - MAE / RMSE for wind speed (ff) and wind gust (gff)  
  - Angular MAE / RMSE for wind direction (dd)  
- **Baselines**:  
  - Auto-regression (AR)  
  - Linear regression (LR)  
  - KNN interpolation  
  - Inverse distance weighting (IDW)  

---





## 📜 Citation
If you use this code, please cite:
```bibtex

```

---

