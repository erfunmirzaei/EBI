from pathlib import Path
import random
import ml_confs

main_path = Path(__file__).parent
data_path = main_path / "__data__"
configs = ml_confs.from_file(main_path / "configs.yaml")