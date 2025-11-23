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
import sys

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chemical Process Analysis API", version="1.0.0")

# Store pristine defaults at startup
DEFAULT_PARAMS = None
MULTIPLIER_DATA = None
PROJECT_DATA = None

def log_error_with_details(error_msg, exception=None, extra_context=None):
    """Comprehensive error logging with context"""
    logger.error(f"🚨 {error_msg}")
    if exception:
        logger.error(f"Exception type: {type(exception).__name__}")
        logger.error(f"Exception message: {str(exception)}")
        logger.error("Stack trace:")
        logger.error(traceback.format_exc())
    if extra_context:
        logger.error(f"Context: {extra_context}")

def load_data_files():
    """Load all required data files at startup with comprehensive error handling"""
    global DEFAULT_PARAMS, MULTIPLIER_DATA, PROJECT_DATA
    
    try:
        logger.info("🔄 Starting data file loading process...")
        
        # Load default parameters
        DEFAULT_PARAMS = deepcopy(model.PARAMS)
        logger.info(f"✅ Loaded default parameters: {len(DEFAULT_PARAMS)} parameters")
        logger.debug(f"Default PARAMS sample: {dict(list(DEFAULT_PARAMS.items())[:5])}")
        
        # Load multiplier data
        multiplier_files = ["sectorwise_multipliers.csv", "multiplier_data.csv"]
        multiplier_loaded = False
        
        for multiplier_file in multiplier_files:
            multiplier_path = Path(multiplier_file)
            if multiplier_path.exists():
                try:
                    MULTIPLIER_DATA = pd.read_csv(multiplier_path)
                    logger.info(f"✅ Loaded multiplier data from {multiplier_file}")
                    logger.info(f"Multiplier data shape: {MULTIPLIER_DATA.shape}")
                    logger.debug(f"Multiplier columns: {list(MULTIPLIER_DATA.columns)}")
                    multiplier_loaded = True
                    break
                except Exception as e:
                    log_error_with_details(f"Failed to load {multiplier_file}", e)
                    continue
        
        if not multiplier_loaded:
            raise FileNotFoundError(f"Could not load multiplier data from any of: {multiplier_files}")
        
        # Load project data
        project_path = Path("project_data.csv")
        if project_path.exists():
            try:
                PROJECT_DATA = pd.read_csv(project_path)
                logger.info(f"✅ Loaded project data from project_data.csv")
                logger.info(f"Project data shape: {PROJECT_DATA.shape}")
                logger.debug(f"Project data columns: {list(PROJECT_DATA.columns)}")
                logger.debug(f"Available countries: {PROJECT_DATA['Country'].unique()}")
                logger.debug(f"Available products: {PROJECT_DATA['Main_Prod'].unique()}")
            except Exception as e:
                log_error_with_details("Failed to load project_data.csv", e)
                raise
        else:
            raise FileNotFoundError("project_data.csv not found")
            
        logger.info("🎉 All data files loaded successfully!")
        
    except Exception as e:
        log_error_with_details("CRITICAL: Failed to load data files", e)
        raise

@contextmanager
def request_context(request_id: str = None):
    """Context manager to handle request-specific state with comprehensive logging"""
    request_id = request_id or f"req_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    logger.info(f"🔹 Starting request {request_id}")
    
    try:
        # Reset to defaults at start of each request
        original_params = getattr(model, 'PARAMS', {})
        model.PARAMS = deepcopy(DEFAULT_PARAMS)
        logger.debug(f"✅ Reset PARAMS to defaults for request {request_id}")
        logger.debug(f"Construction period after reset: {model.PARAMS['construction_prd']}")
        
        # Create fresh copies of data
        multiplier_data = MULTIPLIER_DATA.copy() if MULTIPLIER_DATA is not None else None
        project_data = PROJECT_DATA.copy() if PROJECT_DATA is not None else None
        
        logger.debug(f"Created fresh data copies for request {request_id}")
        
        yield {
            "PARAMS": model.PARAMS,
            "sectorwise_multipliers": multiplier_data,
            "project_data": project_data,
            "request_id": request_id
        }
        
        logger.info(f"✅ Request {request_id} completed successfully")
        
    except Exception as e:
        log_error_with_details(f"Request {request_id} context setup failed", e, {
            "original_params": original_params,
            "default_params": DEFAULT_PARAMS
        })
        raise HTTPException(status_code=500, detail="Internal server error during request setup")
    finally:
        # Always restore original PARAMS to avoid state leakage
        if 'original_params' in locals():
            model.PARAMS = original_params
            logger.debug(f"Restored original PARAMS after request {request_id}")

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
    logger.info("🚀 Starting FastAPI application...")
    try:
        load_data_files()
        logger.info("✅ FastAPI startup completed successfully")
    except Exception as e:
        log_error_with_details("CRITICAL: FastAPI startup failed", e)
        # Don't raise here to allow the app to start, but it will fail on first request

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Chemical Process Analysis API is running",
        "data_loaded": all([DEFAULT_PARAMS is not None, MULTIPLIER_DATA is not None, PROJECT_DATA is not None])
    }

@app.get("/debug/params")
async def debug_params():
    """Debug endpoint to check current parameters"""
    return {
        "default_params": DEFAULT_PARAMS,
        "current_params": model.PARAMS,
        "multiplier_data_loaded": MULTIPLIER_DATA is not None,
        "project_data_loaded": PROJECT_DATA is not None
    }

@app.post("/run_analysis")
async def run_analysis(request: AnalysisRequest):
    """Main analysis endpoint with comprehensive error handling"""
    request_id = f"analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    logger.info(f"🎯 Starting analysis request {request_id}")
    logger.info(f"Request parameters: location={request.location}, product={request.product}, "
                f"plant_mode={request.plant_mode}, fund_mode={request.fund_mode}")
    
    with request_context(request_id) as context:
        try:
            # Log parameter state at start
            logger.debug(f"PARAMS at request start - construction_prd: {model.PARAMS['construction_prd']}")
            
            # Validate we have the required data
            if MULTIPLIER_DATA is None or PROJECT_DATA is None:
                error_msg = "Data files not loaded properly"
                log_error_with_details(error_msg, extra_context={
                    "multiplier_data_loaded": MULTIPLIER_DATA is not None,
                    "project_data_loaded": PROJECT_DATA is not None
                })
                raise HTTPException(status_code=500, detail=error_msg)
            
            # Log parameter updates
            params_updated = []
            if request.operating_prd is not None:
                model.PARAMS['operating_prd'] = request.operating_prd
                params_updated.append(f"operating_prd: {request.operating_prd}")
            if request.construction_prd is not None:
                model.PARAMS['construction_prd'] = request.construction_prd
                params_updated.append(f"construction_prd: {request.construction_prd}")
            if request.util_fac_year1 is not None:
                model.PARAMS['util_fac_year1'] = request.util_fac_year1
                params_updated.append(f"util_fac_year1: {request.util_fac_year1}")
            if request.util_fac_year2 is not None:
                model.PARAMS['util_fac_year2'] = request.util_fac_year2
                params_updated.append(f"util_fac_year2: {request.util_fac_year2}")
            if request.util_fac_remaining is not None:
                model.PARAMS['util_fac_remaining'] = request.util_fac_remaining
                params_updated.append(f"util_fac_remaining: {request.util_fac_remaining}")
            if request.infl is not None:
                model.PARAMS['Infl'] = request.infl
                params_updated.append(f"infl: {request.infl}")
            if request.RR is not None:
                model.PARAMS['RR'] = request.RR
                params_updated.append(f"RR: {request.RR}")
            if request.IRR is not None:
                model.PARAMS['IRR'] = request.IRR
                params_updated.append(f"IRR: {request.IRR}")
            if request.capex_spread is not None:
                model.PARAMS['capex_spread'] = request.capex_spread
                params_updated.append(f"capex_spread: {request.capex_spread}")
            if request.shrDebt is not None:
                model.PARAMS['shrDebt'] = request.shrDebt
                params_updated.append(f"shrDebt: {request.shrDebt}")
            if request.ownerCost is not None:
                model.PARAMS['OwnerCost'] = request.ownerCost
                params_updated.append(f"ownerCost: {request.ownerCost}")
            if request.credit is not None:
                model.PARAMS['credit'] = request.credit
                params_updated.append(f"credit: {request.credit}")
            if request.PRIcoef is not None:
                model.PARAMS['PRIcoef'] = request.PRIcoef
                params_updated.append(f"PRIcoef: {request.PRIcoef}")
            if request.CONcoef is not None:
                model.PARAMS['CONcoef'] = request.CONcoef
                params_updated.append(f"CONcoef: {request.CONcoef}")
            if request.EcNatGas is not None:
                model.PARAMS['EcNatGas'] = request.EcNatGas
                params_updated.append(f"EcNatGas: {request.EcNatGas}")
            if request.ngCcontnt is not None:
                model.PARAMS['ngCcontnt'] = request.ngCcontnt
                params_updated.append(f"ngCcontnt: {request.ngCcontnt}")
            if request.eEFF is not None:
                model.PARAMS['eEFF'] = request.eEFF
                params_updated.append(f"eEFF: {request.eEFF}")
            if request.elEFF is not None:
                model.PARAMS['elEFF'] = request.elEFF
                params_updated.append(f"elEFF: {request.elEFF}")
            if request.hEFF is not None:
                model.PARAMS['hEFF'] = request.hEFF
                params_updated.append(f"hEFF: {request.hEFF}")

            if params_updated:
                logger.info(f"Updated parameters: {', '.join(params_updated)}")
            
            # Filter project data for this request
            logger.info(f"Filtering project data for location: '{request.location}', product: '{request.product}'")
            
            project_data = PROJECT_DATA[
                (PROJECT_DATA['Country'] == request.location) & 
                (PROJECT_DATA['Main_Prod'] == request.product)
            ]
            
            logger.info(f"Found {len(project_data)} matching project records")
            
            if len(project_data) == 0:
                available_locations = PROJECT_DATA['Country'].unique()
                available_products = PROJECT_DATA['Main_Prod'].unique()
                
                error_msg = f"No project data found for location '{request.location}' and product '{request.product}'"
                log_error_with_details(error_msg, extra_context={
                    "available_locations": list(available_locations),
                    "available_products": list(available_products)
                })
                
                raise HTTPException(
                    status_code=404,
                    detail={
                        "message": error_msg,
                        "available_locations": list(available_locations),
                        "available_products": list(available_products)
                    }
                )

            # Update project data with any provided overrides
            overrides_applied = []
            for idx in project_data.index:
                if request.baseYear is not None:
                    project_data.at[idx, 'Base_Yr'] = request.baseYear
                    overrides_applied.append(f"Base_Yr: {request.baseYear}")
                if request.corpTAX is not None:
                    project_data.at[idx, 'corpTAX'] = request.corpTAX
                    overrides_applied.append(f"corpTAX: {request.corpTAX}")
                if request.Feed_Price is not None:
                    project_data.at[idx, 'Feed_Price'] = request.Feed_Price
                    overrides_applied.append(f"Feed_Price: {request.Feed_Price}")
                if request.Fuel_Price is not None:
                    project_data.at[idx, 'Fuel_Price'] = request.Fuel_Price
                    overrides_applied.append(f"Fuel_Price: {request.Fuel_Price}")
                if request.Elect_Price is not None:
                    project_data.at[idx, 'Elect_Price'] = request.Elect_Price
                    overrides_applied.append(f"Elect_Price: {request.Elect_Price}")
                if request.CO2price is not None:
                    project_data.at[idx, 'CO2price'] = request.CO2price
                    overrides_applied.append(f"CO2price: {request.CO2price}")
                if request.CAPEX is not None:
                    project_data.at[idx, 'CAPEX'] = request.CAPEX
                    overrides_applied.append(f"CAPEX: {request.CAPEX}")
                if request.OPEX is not None:
                    project_data.at[idx, 'OPEX'] = request.OPEX
                    overrides_applied.append(f"OPEX: {request.OPEX}")
                if request.Cap is not None:
                    project_data.at[idx, 'Cap'] = request.Cap
                    overrides_applied.append(f"Cap: {request.Cap}")
                if request.Yld is not None:
                    project_data.at[idx, 'Yld'] = request.Yld
                    overrides_applied.append(f"Yld: {request.Yld}")
                if request.feedEcontnt is not None:
                    project_data.at[idx, 'feedEcontnt'] = request.feedEcontnt
                    overrides_applied.append(f"feedEcontnt: {request.feedEcontnt}")
                if request.Heat_req is not None:
                    project_data.at[idx, 'Heat_req'] = request.Heat_req
                    overrides_applied.append(f"Heat_req: {request.Heat_req}")
                if request.Elect_req is not None:
                    project_data.at[idx, 'Elect_req'] = request.Elect_req
                    overrides_applied.append(f"Elect_req: {request.Elect_req}")
                if request.feedCcontnt is not None:
                    project_data.at[idx, 'feedCcontnt'] = request.feedCcontnt
                    overrides_applied.append(f"feedCcontnt: {request.feedCcontnt}")

            if overrides_applied:
                logger.info(f"Applied project data overrides: {', '.join(set(overrides_applied))}")

            # Run the analysis
            logger.info("🚀 Calling Analytics_Model2...")
            
            try:
                results = model.Analytics_Model2(
                    multiplier=MULTIPLIER_DATA,
                    project_data=project_data,
                    location=request.location,
                    product=request.product,
                    plant_mode=request.plant_mode,
                    fund_mode=request.fund_mode,
                    opex_mode=request.opex_mode,
                    carbon_value=request.carbon_value
                )
                
                logger.info(f"✅ Analysis completed successfully. Results shape: {results.shape}")
                logger.debug(f"Results columns: {list(results.columns)}")
                
                # Convert results to list of dicts for JSON response
                response_data = results.to_dict(orient='records')
                
                logger.info(f"📊 Returning {len(response_data)} records for request {request_id}")
                
                return {
                    "success": True,
                    "request_id": request_id,
                    "data": response_data,
                    "metadata": {
                        "records_count": len(response_data),
                        "columns": list(results.columns)
                    }
                }
                
            except Exception as model_error:
                log_error_with_details("Analytics_Model2 execution failed", model_error, {
                    "input_parameters": {
                        "location": request.location,
                        "product": request.product,
                        "plant_mode": request.plant_mode,
                        "fund_mode": request.fund_mode,
                        "opex_mode": request.opex_mode,
                        "carbon_value": request.carbon_value
                    },
                    "project_data_shape": project_data.shape,
                    "multiplier_data_shape": MULTIPLIER_DATA.shape if MULTIPLIER_DATA is not None else None
                })
                raise HTTPException(status_code=500, detail=f"Model execution failed: {str(model_error)}")

        except HTTPException:
            # Re-raise HTTP exceptions as they are
            raise
        except Exception as e:
            log_error_with_details(f"Unexpected error in run_analysis for request {request_id}", e, {
                "request_data": request.dict(),
                "current_params": model.PARAMS
            })
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to catch any unhandled exceptions"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    log_error_with_details(f"Global exception handler caught error for request {request_id}", exc, {
        "request_method": request.method,
        "request_url": str(request.url),
        "client_host": request.client.host if request.client else "unknown"
    })
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "request_id": request_id,
            "detail": "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    try:
        import uvicorn
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_config=None,  # Use our custom logging
            access_log=False  # We handle logging ourselves
        )
    except Exception as e:
        log_error_with_details("Uvicorn server failed to start", e)
        sys.exit(1)
