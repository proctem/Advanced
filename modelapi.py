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
            logger.info(f"Loaded project data with columns: {PROJECT_DATA.columns.tolist()}")
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

@app.post("/run_analysis")
async def run_analysis(request: AnalysisRequest):
    try:
        # Validate we have the required data
        if MULTIPLIER_DATA is None or PROJECT_DATA is None:
            raise HTTPException(status_code=500, detail="Data files not loaded")
        
        logger.info(f"Processing request for {request.location}/{request.product}")

        # Update model parameters from request
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

        # Handle custom projects
        requested_product = str(request.product).strip().lower()
        is_custom = requested_product == "custom"
        
        if is_custom:
            # Try to find country-specific custom template first
            custom_filter = (
                (PROJECT_DATA['Country'].str.strip() == request.location) & 
                (PROJECT_DATA['Main_Prod'].str.strip().str.lower() == "custom")
            )
            project_data = PROJECT_DATA[custom_filter].copy()
            
            # If none found, use any custom template
            if len(project_data) == 0:
                project_data = PROJECT_DATA[
                    PROJECT_DATA['Main_Prod'].str.strip().str.lower() == "custom"
                ].copy()
            
            # If still empty, create from request parameters
            if len(project_data) == 0:
                logger.warning("Creating custom project from request parameters")
                project_data = pd.DataFrame([{
                    'Country': request.location,
                    'ProcTech': 'Custom',
                    'Feedstock': 'Custom',
                    'Main_Prod': 'Custom',
                    'Plant_Size': request.plant_size,
                    'Plant_Effy': request.plant_effy,
                    'Cap': request.Cap,
                    'Yld': request.Yld,
                    'Base_Yr': request.baseYear,
                    'CAPEX': request.CAPEX,
                    'OPEX': request.OPEX,
                    'Feed_Price': request.Feed_Price,
                    'Heat_req': request.Heat_req,
                    'Elect_req': request.Elect_req,
                    'Fuel_Price': request.Fuel_Price,
                    'Elect_Price': request.Elect_Price,
                    'feedEcontnt': request.feedEcontnt,
                    'feedCcontnt': request.feedCcontnt,
                    'corpTAX': request.corpTAX,
                    'CO2price': request.CO2price
                }])
        else:
            # Handle standard products
            project_data = PROJECT_DATA[
                (PROJECT_DATA['Country'] == request.location) & 
                (PROJECT_DATA['Main_Prod'] == request.product)
            ].copy()

        # Validate we got project data
        if len(project_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No project data found for location '{request.location}' and product '{request.product}'"
            )

        # Update project data with any provided overrides
        for idx in project_data.index:
            for param in ['Cap', 'Yld', 'Base_Yr', 'CAPEX', 'OPEX', 'Feed_Price',
                         'Heat_req', 'Elect_req', 'Fuel_Price', 'Elect_Price',
                         'feedEcontnt', 'feedCcontnt', 'corpTAX', 'CO2price']:
                if hasattr(request, param) and getattr(request, param) is not None:
                    project_data.at[idx, param] = getattr(request, param)

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
            plant_size=request.plant_size,
            plant_effy=request.plant_effy
        )
        
        return results.to_dict(orient='records')

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
