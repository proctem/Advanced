from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import originalmodela as model
import pandas as pd
import numpy as np
import logging
from copy import deepcopy
from pathlib import Path
import json
import traceback
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Store pristine defaults at startup
DEFAULT_PARAMS = None
MULTIPLIER_DATA = None

def load_data_files():
    """Load required data files at startup (only multipliers now)"""
    global DEFAULT_PARAMS, MULTIPLIER_DATA
    
    try:
        # Load default parameters
        DEFAULT_PARAMS = deepcopy(model.PARAMS)
        logger.info("Loaded default parameters")
        
        # Load multiplier data
        multiplier_path = Path("sectorwise_multipliers.csv")
        if multiplier_path.exists():
            MULTIPLIER_DATA = pd.read_csv(multiplier_path)
            logger.info("Loaded multiplier data")
        else:
            raise FileNotFoundError("multiplier_data.csv not found")
            
    except Exception as e:
        logger.critical(f"Failed to load data files: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize application data"""
    try:
        load_data_files()
        logger.info("Application startup completed")
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")
        raise RuntimeError("Application failed to initialize")

class ProjectData(BaseModel):
    # Project-specific parameters (all optional with defaults from original model)
    baseYear: Optional[int] = None
    corpTAX: Optional[float] = None
    Feed_Price: Optional[float] = None
    Fuel_Price: Optional[float] = None
    Elect_Price: Optional[float] = None
    CO2price: Optional[float] = None
    CAPEX: Optional[float] = None
    OPEX: Optional[float] = None
    Cap: Optional[float] = None
    Yld: Optional[float] = None
    feedEcontnt: Optional[float] = None
    Heat_req: Optional[float] = None
    Elect_req: Optional[float] = None
    feedCcontnt: Optional[float] = None
    Plant_Size: Optional[str] = None  # Added this field
    Plant_Effy: Optional[str] = None  # Added this field

class AnalysisRequest(BaseModel):
    # Required parameters
    location: str
    product: str
    plant_mode: str  # "Green" or "Brown"
    fund_mode: str   # "Debt", "Equity", or "Mixed"
    
    # Project data - all fields optional
    project_data: Optional[ProjectData] = None
    
    # Optional parameters with defaults
    opex_mode: Optional[str] = "Inflated"
    plant_size: Optional[str] = "Large"
    plant_effy: Optional[str] = "High"
    carbon_value: Optional[str] = "No"
    operating_prd: Optional[int] = None
    util_fac_year1: Optional[float] = None
    util_fac_year2: Optional[float] = None
    util_fac_remaining: Optional[float] = None
    infl: Optional[float] = None
    RR: Optional[float] = None
    IRR: Optional[float] = None
    construction_prd: Optional[int] = None
    capex_spread: Optional[List[float]] = None  # [yr1, yr2, yr3]
    shrDebt: Optional[float] = None
    ownerCost: Optional[float] = None
    credit: Optional[float] = None
    PRIcoef: Optional[float] = None
    CONcoef: Optional[float] = None
    EcNatGas: Optional[float] = None
    ngCcontnt: Optional[float] = None
    eEFF: Optional[float] = None
    elEFF: Optional[float] = None
    hEFF: Optional[float] = None

@app.post("/run_analysis")
async def run_analysis(request: AnalysisRequest):
    try:
        # Validate we have the required data
        if MULTIPLIER_DATA is None:
            raise HTTPException(status_code=500, detail="Multiplier data not loaded")
        
        # Create a project data dictionary with defaults or provided values
        project_data = {
            'Country': request.location,
            'Main_Prod': request.product,
            'Base_Yr': request.project_data.baseYear if request.project_data and request.project_data.baseYear else None,
            'corpTAX': request.project_data.corpTAX if request.project_data and request.project_data.corpTAX else None,
            'Feed_Price': request.project_data.Feed_Price if request.project_data and request.project_data.Feed_Price else None,
            'Fuel_Price': request.project_data.Fuel_Price if request.project_data and request.project_data.Fuel_Price else None,
            'Elect_Price': request.project_data.Elect_Price if request.project_data and request.project_data.Elect_Price else None,
            'CO2price': request.project_data.CO2price if request.project_data and request.project_data.CO2price else None,
            'CAPEX': request.project_data.CAPEX if request.project_data and request.project_data.CAPEX else None,
            'OPEX': request.project_data.OPEX if request.project_data and request.project_data.OPEX else None,
            'Cap': request.project_data.Cap if request.project_data and request.project_data.Cap else None,
            'Yld': request.project_data.Yld if request.project_data and request.project_data.Yld else None,
            'feedEcontnt': request.project_data.feedEcontnt if request.project_data and request.project_data.feedEcontnt else None,
            'Heat_req': request.project_data.Heat_req if request.project_data and request.project_data.Heat_req else None,
            'Elect_req': request.project_data.Elect_req if request.project_data and request.project_data.Elect_req else None,
            'feedCcontnt': request.project_data.feedCcontnt if request.project_data and request.project_data.feedCcontnt else None,
            'Plant_Size': request.plant_size,  # Added this field from request
            'Plant_Effy': request.plant_effy   # Added this field from request
        }
        
        # Convert to DataFrame (single row)
        project_df = pd.DataFrame([project_data])

        # Fill any remaining NaN values with defaults
        project_df.fillna({
            'Plant_Size': 'Large',
            'Plant_Effy': 'High'
        }, inplace=True)

        # Update model parameters from request (only if provided)
        if request.operating_prd is not None:
            model.PARAMS['operating_prd'] = request.operating_prd
        if request.construction_prd is not None:
            model.PARAMS['construction_prd'] = request.construction_prd
        if request.util_fac_year1 is not None:
            model.PARAMS['util_fac_year1'] = request.util_fac_year1
        if request.util_fac_year2 is not None:
            model.PARAMS['util_fac_year2'] = request.util_fac_year2
        if request.util_fac_remaining is not None:
            model.PARAMS['util_fac_remaining'] = request.util_fac_remaining
        if request.infl is not None:
            model.PARAMS['Infl'] = request.infl
        if request.RR is not None:
            model.PARAMS['RR'] = request.RR
        if request.IRR is not None:
            model.PARAMS['IRR'] = request.IRR
        if request.capex_spread is not None:
            model.PARAMS['capex_spread'] = request.capex_spread
        if request.shrDebt is not None:
            model.PARAMS['shrDebt'] = request.shrDebt
        if request.ownerCost is not None:
            model.PARAMS['OwnerCost'] = request.ownerCost
        if request.credit is not None:
            model.PARAMS['credit'] = request.credit
        if request.PRIcoef is not None:
            model.PARAMS['PRIcoef'] = request.PRIcoef
        if request.CONcoef is not None:
            model.PARAMS['CONcoef'] = request.CONcoef
        if request.EcNatGas is not None:
            model.PARAMS['EcNatGas'] = request.EcNatGas
        if request.ngCcontnt is not None:
            model.PARAMS['ngCcontnt'] = request.ngCcontnt
        if request.eEFF is not None:
            model.PARAMS['eEFF'] = request.eEFF
        if request.elEFF is not None:
            model.PARAMS['elEFF'] = request.elEFF
        if request.hEFF is not None:
            model.PARAMS['hEFF'] = request.hEFF

        # Run the analysis
        results = model.Analytics_Model2(
            multiplier=MULTIPLIER_DATA,
            project_data=project_df,
            location=request.location,
            product=request.product,
            plant_mode=request.plant_mode,
            fund_mode=request.fund_mode,
            opex_mode=request.opex_mode,
            carbon_value=request.carbon_value,
            plant_size=request.plant_size,
            plant_effy=request.plant_effy
        )
        
        # Convert results to list of dicts for JSON response
        return results.to_dict(orient='records')

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
