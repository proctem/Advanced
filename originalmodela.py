from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import uvicorn
import logging
from originalmodel import Analytics_Model2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Project Economics Model API",
    description="API for chemical plant economics analysis - Strict Payload Only",
    version="2.0.0"
)

class AnalysisRequest(BaseModel):
    # Required parameters with no defaults
    location: str
    plant_effy: Optional[str] = None
    plant_size: Optional[str] = None
    plant_mode: str
    fund_mode: str
    opex_mode: str
    carbon_value: str
    
    # Optional parameters
    product: Optional[str] = None
    
    # Optional technical parameters
    operating_prd: int
    util_operating_first: float
    util_operating_second: float
    util_operating_third: float
    infl: float
    RR: float
    IRR: float
    construction_prd: int
    capex_spread: List[float]
    shrDebt_value: float
    baseYear: int
    ownerCost: float
    corpTAX_value: float
    Feed_Price: float
    Fuel_Price: float
    Elect_Price: float
    CarbonTAX_value: float
    credit_value: float
    CAPEX: float
    OPEX: float
    PRIcoef: float
    CONcoef: float
    
    # Technical parameters
    EcNatGas: float
    ngCcontnt: float
    eEFF: float
    hEFF: float
    Cap: float
    Yld: float
    feedEcontnt: float
    Heat_req: float
    Elect_req: float
    feedCcontnt: float

@app.on_event("startup")
async def startup_event():
    """Load required data files"""
    global project_datas, multipliers
    try:
        project_datas = pd.read_csv("./project_data.csv")
        multipliers = pd.read_csv("./sectorwise_multipliers.csv")
        logger.info("Data files loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Required data files not found: {str(e)}")
        raise Exception(f"Required data files not found: {str(e)}")

@app.post("/analyze", response_model=List[dict])
async def run_analysis(request: AnalysisRequest):
    """
    Run economic analysis using ONLY the provided payload values.
    All parameters are required except product, plant_size, and plant_effy - no defaults will be used.
    """
    # Convert request to dict and log everything
    config = request.dict()
    logger.info("\n=== PAYLOAD VALUES RECEIVED ===")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    # Validate parameters
    validate_parameters(config)
    
    # Create data row from payload only
    custom_data = create_custom_data_row(config)
    
    # Run analysis
    try:
        logger.info("Starting analysis with payload values only...")
        results = Analytics_Model2(
            multiplier=multipliers,
            project_data=custom_data,
            location=config["location"],
            product=config.get("product", ""),  # Use empty string if product not provided
            plant_mode=config["plant_mode"],
            fund_mode=config["fund_mode"],
            opex_mode=config["opex_mode"],
            plant_size=config.get("plant_size", ""),  # Use empty string if plant_size not provided
            plant_effy=config.get("plant_effy", ""),  # Use empty string if plant_effy not provided
            carbon_value=config["carbon_value"]
        )
        
        logger.info("Analysis completed successfully")
        return results.to_dict(orient='records')
    
    except Exception as e:
        logger.error(f"Error running analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error running analysis: {str(e)}")

def validate_parameters(config: dict):
    """Validate all payload parameters"""
    if config["location"] not in project_datas['Country'].unique():
        logger.error(f"Invalid location: {config['location']}")
        raise HTTPException(status_code=400, detail="Invalid location")
    
    # Only validate product if it's provided
    if "product" in config and config["product"] is not None:
        if config["product"] not in project_datas['Main_Prod'].unique():
            logger.error(f"Invalid product: {config['product']}")
            raise HTTPException(status_code=400, detail="Invalid product")
    
    # Only validate plant_size if it's provided
    if "plant_size" in config and config["plant_size"] is not None:
        if config["plant_size"] not in ["Large", "Small"]:
            raise HTTPException(status_code=400, detail="plant_size must be 'Large' or 'Small'")
    
    # Only validate plant_effy if it's provided
    if "plant_effy" in config and config["plant_effy"] is not None:
        if config["plant_effy"] not in ["High", "Low"]:
            raise HTTPException(status_code=400, detail="plant_effy must be 'High' or 'Low'")
    
    if config["plant_mode"] not in ["Green", "Brown"]:
        raise HTTPException(status_code=400, detail="plant_mode must be 'Green' or 'Brown'")
    
    if config["fund_mode"] not in ["Debt", "Equity", "Mixed"]:
        raise HTTPException(status_code=400, detail="fund_mode must be 'Debt', 'Equity', or 'Mixed'")
    
    if config["opex_mode"] not in ["Inflated", "Uninflated"]:
        raise HTTPException(status_code=400, detail="opex_mode must be 'Inflated' or 'Uninflated'")
    
    if config["carbon_value"] not in ["Yes", "No"]:
        raise HTTPException(status_code=400, detail="carbon_value must be 'Yes' or 'No'")
    
    if sum(config["capex_spread"]) != 1.0:
        raise HTTPException(status_code=400, detail="capex_spread values must sum to 1.0")
    
    if config["eEFF"] <= 0 or config["eEFF"] > 1:
        raise HTTPException(status_code=400, detail="Electrical efficiency must be between 0 and 1")
    
    if config["hEFF"] <= 0 or config["hEFF"] > 1:
        raise HTTPException(status_code=400, detail="Heat efficiency must be between 0 and 1")

def create_custom_data_row(config: dict) -> pd.DataFrame:
    """Create data row from payload values only"""
    data = {
        "Country": config["location"],
        "Main_Prod": config.get("product", ""),  # Use empty string if product not provided
        "Plant_Size": config.get("plant_size", ""),  # Use empty string if plant_size not provided
        "Plant_Effy": config.get("plant_effy", ""),  # Use empty string if plant_effy not provided
        "ProcTech": "Custom",
        "Base_Yr": config["baseYear"],
        "Cap": config["Cap"],
        "Yld": config["Yld"],
        "feedEcontnt": config["feedEcontnt"],
        "feedCcontnt": config["feedCcontnt"],
        "Heat_req": config["Heat_req"],
        "Elect_req": config["Elect_req"],
        "Feed_Price": config["Feed_Price"],
        "Fuel_Price": config["Fuel_Price"],
        "Elect_Price": config["Elect_Price"],
        "CO2price": config["CarbonTAX_value"],
        "corpTAX": config["corpTAX_value"],
        "CAPEX": config["CAPEX"],
        "OPEX": config["OPEX"],
        "EcNatGas": config["EcNatGas"],
        "ngCcontnt": config["ngCcontnt"],
        "eEFF": config["eEFF"],
        "hEFF": config["hEFF"]
    }
    
    logger.info("\nCustom Data Row Created From Payload:")
    for key, value in data.items():
        logger.info(f"{key}: {value}")
    
    return pd.DataFrame([data])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
