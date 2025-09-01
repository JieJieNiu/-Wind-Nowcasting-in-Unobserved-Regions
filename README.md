# 	Wind Nowcasting in Unobserved Regions
Repository Structure
.
├── data/                # Raw and processed KNMI weather data
│   ├── stations.csv
│   ├── virtual_nodes.csv
│   └── ...
├── models/              # GNN model definitions
│   ├── temporal_gcn.py
│   ├── moco_module.py
│   └── ...
├── train_loop.py        # Main training pipeline
├── test.py              # Evaluation scripts
├── utils/               # Helper functions (graph diffusion, loaders, metrics)
├── visulization/        # Scripts for plots & figures
└── README.md            # This file
