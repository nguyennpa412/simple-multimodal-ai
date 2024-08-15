import os
import warnings

from src.utils.resources import Resources
from src.components.main import Main
from src.utils.loader import AppLoader

warnings.filterwarnings('ignore')

file_path = os.path.abspath(__file__)
home_dir = os.path.dirname(file_path)
os.environ["APP_HOME_DIR"] = home_dir
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    # LOAD CONFIG
    app_config_path = f"{home_dir}/configs/app_config.yaml"
    app_loader = AppLoader(app_config_path=app_config_path)
    app_loader.load_config()
    
    # CHECK REQUIRED RESOURCES
    resources = Resources(speakers_type=app_loader.speakers_type)
    resources.check_required_files()
    
    # LOAD MODULES
    app_loader.load_modules()

    # RUN APP
    main = Main(app_loader=app_loader)
    main_component = main.get_component()
    main_component.launch(**app_loader.launch_config)
