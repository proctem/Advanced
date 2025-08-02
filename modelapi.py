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
PROJECT_DATA = None

def load_data_files():
    """Load all required data files at startup"""
    global DEFAULT_PARAMS, MULTIPLIER_DATA, PROJECT_DATA
    
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
        
        # Load project data
        project_path = Path("project_data.csv")
        if project_path.exists():
            PROJECT_DATA = pd.read_csv(project_path)
            logger.info("Loaded project data")
        else:
            raise FileNotFoundError("project_data.csv not found")
            
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

@contextmanager
def request_context():
    """Context manager to handle request-specific state"""
    try:
        # Reset to defaults at start of each request
        model.PARAMS = deepcopy(DEFAULT_PARAMS)
        logger.debug("Reset PARAMS to defaults")
        
        # Create fresh copies of data
        multiplier_data = MULTIPLIER_DATA.copy()
        project_data = PROJECT_DATA.copy()
        logger.debug("Created fresh data copies")
        
        yield {
            "PARAMS": model.PARAMS,
            "multiplier_data": multiplier_data,
            "project_data": project_data
        }
        
    except Exception as e:
        logger.error(f"Request context setup failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")


class AnalysisRequest(BaseModel):
    # Required parameters
    location: str
    product: str
    plant_mode: str  # "Green" or "Brown"
    fund_mode: str   # "Debt", "Equity", or "Mixed"
    
    # Optional parameters with defaults
    opex_mode: Optional[str] = "Inflated"
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
    baseYear: Optional[int] = None
    ownerCost: Optional[float] = None
    corpTAX: Optional[float] = None
    CO2price: Optional[float] = None
    Feed_Price: Optional[float] = None
    Fuel_Price: Optional[float] = None
    Elect_Price: Optional[float] = None
    credit: Optional[float] = None
    CAPEX: Optional[float] = None
    OPEX: Optional[float] = None
    PRIcoef: Optional[float] = None
    CONcoef: Optional[float] = None
    EcNatGas: Optional[float] = None
    ngCcontnt: Optional[float] = None
    eEFF: Optional[float] = None
    elEFF: Optional[float] = None
    hEFF: Optional[float] = None
    Cap: Optional[float] = None
    Yld: Optional[float] = None
    feedEcontnt: Optional[float] = None
    Heat_req: Optional[float] = None
    Elect_req: Optional[float] = None
    feedCcontnt: Optional[float] = None

@app.on_event("startup")
async def startup_event():
    """Load data files when starting the application"""
    load_data_files()
# Add this to your API code (after loading DEFAULT_PARAMS)
DEFAULT_PARAMS = deepcopy(model.PARAMS)  # Store pristine defaults at startup

@app.post("/run_analysis")
async def run_analysis(request: AnalysisRequest):
    print(f"Current PARAMS at start: {model.PARAMS['construction_prd']}")  # Debug log
    try:
        # Validate we have the required data
        if MULTIPLIER_DATA is None or PROJECT_DATA is None:
            raise HTTPException(status_code=500, detail="Data files not loaded")
        
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
            model.PARAMS['elEFF'] = request.elEFF  # Assuming same as eEFF
        if request.hEFF is not None:
            model.PARAMS['hEFF'] = request.hEFF

        # Filter project data for this request
        project_data = PROJECT_DATA[
            (PROJECT_DATA['Country'] == request.location) & 
            (PROJECT_DATA['Main_Prod'] == request.product)
        ]
        
        if len(project_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No project data found for location '{request.location}' and product '{request.product}'"
            )

        # Update project data with any provided overrides
        for idx in project_data.index:
            if request.baseYear is not None:
                project_data.at[idx, 'Base_Yr'] = request.baseYear
            if request.corpTAX is not None:
                project_data.at[idx, 'corpTAX'] = request.corpTAX
            if request.Feed_Price is not None:
                project_data.at[idx, 'Feed_Price'] = request.Feed_Price
            if request.Fuel_Price is not None:
                project_data.at[idx, 'Fuel_Price'] = request.Fuel_Price
            if request.Elect_Price is not None:
                project_data.at[idx, 'Elect_Price'] = request.Elect_Price
            if request.CO2price is not None:
                project_data.at[idx, 'CO2price'] = request.CO2price
            if request.CAPEX is not None:
                project_data.at[idx, 'CAPEX'] = request.CAPEX
            if request.OPEX is not None:
                project_data.at[idx, 'OPEX'] = request.OPEX
            if request.Cap is not None:
                project_data.at[idx, 'Cap'] = request.Cap
            if request.Yld is not None:
                project_data.at[idx, 'Yld'] = request.Yld
            if request.feedEcontnt is not None:
                project_data.at[idx, 'feedEcontnt'] = request.feedEcontnt
            if request.Heat_req is not None:
                project_data.at[idx, 'Heat_req'] = request.Heat_req
            if request.Elect_req is not None:
                project_data.at[idx, 'Elect_req'] = request.Elect_req
            if request.feedCcontnt is not None:
                project_data.at[idx, 'feedCcontnt'] = request.feedCcontnt

        # Run the analysis
        results = model.Analytics_Model2(
            multiplier=MULTIPLIER_DATA,
            project_data=project_data,
            location=request.location,
            product=request.product,
            plant_mode=request.plant_mode,
            fund_mode=request.fund_mode,
            opex_mode=request.opex_mode,
            carbon_value=request.carbon_value,
        )
        
        # Convert results to list of dicts for JSON response
        return results.to_dict(orient='records')

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
