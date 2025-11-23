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
import io

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
        
        # Load multiplier data - FIXED FILENAME
        multiplier_path = Path("sectorwise_multipliers.csv")
        if multiplier_path.exists():
            MULTIPLIER_DATA = pd.read_csv(multiplier_path)
            logger.info("Loaded multiplier data")
        else:
            # Also try the alternative name for backward compatibility
            multiplier_path_alt = Path("multiplier_data.csv")
            if multiplier_path_alt.exists():
                MULTIPLIER_DATA = pd.read_csv(multiplier_path_alt)
                logger.info("Loaded multiplier data from multiplier_data.csv")
            else:
                raise FileNotFoundError("Neither sectorwise_multipliers.csv nor multiplier_data.csv found")
        
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

@contextmanager
def capture_model_logs():
    """Capture all logs and outputs from the model"""
    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)
    
    # Create a custom formatter for the captured logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to the model's logger
    model_logger = logging.getLogger('originalmodela')
    original_handlers = model_logger.handlers[:]
    model_logger.addHandler(ch)
    model_logger.setLevel(logging.DEBUG)
    
    # Also capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    
    try:
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr
        yield log_capture_string, captured_stdout, captured_stderr
    finally:
        # Restore original state
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        model_logger.handlers = original_handlers
        ch.close()

def log_request_details(request: AnalysisRequest):
    """Log detailed request information"""
    logger.info("=== ANALYSIS REQUEST DETAILS ===")
    logger.info(f"Location: {request.location}")
    logger.info(f"Product: {request.product}")
    logger.info(f"Plant Mode: {request.plant_mode}")
    logger.info(f"Fund Mode: {request.fund_mode}")
    logger.info(f"Opex Mode: {request.opex_mode}")
    logger.info(f"Carbon Value: {request.carbon_value}")
    
    # Log all provided parameters
    provided_params = {}
    for field, value in request.dict().items():
        if value is not None and field not in ['location', 'product', 'plant_mode', 'fund_mode', 'opex_mode', 'carbon_value']:
            provided_params[field] = value
    
    if provided_params:
        logger.info("Provided parameter overrides:")
        for param, value in provided_params.items():
            logger.info(f"  {param}: {value}")
    else:
        logger.info("No parameter overrides provided - using defaults")
    
    logger.info("=== CURRENT MODEL PARAMS ===")
    for key, value in model.PARAMS.items():
        logger.info(f"  {key}: {value}")

def log_data_details(project_data):
    """Log details about the data being used"""
    logger.info("=== PROJECT DATA DETAILS ===")
    logger.info(f"Number of records: {len(project_data)}")
    if len(project_data) > 0:
        sample_record = project_data.iloc[0]
        logger.info("Sample record fields:")
        for field in project_data.columns:
            logger.info(f"  {field}: {sample_record[field]}")
    
    logger.info("=== MULTIPLIER DATA DETAILS ===")
    if MULTIPLIER_DATA is not None:
        logger.info(f"Multiplier data shape: {MULTIPLIER_DATA.shape}")
        logger.info(f"Multiplier columns: {list(MULTIPLIER_DATA.columns)}")
        unique_countries = MULTIPLIER_DATA['Country'].unique()
        logger.info(f"Available countries in multipliers: {list(unique_countries)}")

def log_function_call_details():
    """Log details about function calls and parameters"""
    logger.info("=== FUNCTION CALL DETAILS ===")
    logger.info(f"Analytics_Model2 function: {model.Analytics_Model2}")
    logger.info(f"ChemProcess_Model function: {model.ChemProcess_Model}")
    logger.info(f"MicroEconomic_Model function: {model.MicroEconomic_Model}")
    logger.info(f"MacroEconomic_Model function: {model.MacroEconomic_Model}")

@app.on_event("startup")
async def startup_event():
    """Load data files when starting the application"""
    load_data_files()

# Add this to your API code (after loading DEFAULT_PARAMS)
DEFAULT_PARAMS = deepcopy(model.PARAMS)  # Store pristine defaults at startup

@app.post("/run_analysis")
async def run_analysis(request: AnalysisRequest):
    logger.info("=== STARTING NEW ANALYSIS REQUEST ===")
    
    try:
        # Validate we have the required data
        if MULTIPLIER_DATA is None or PROJECT_DATA is None:
            error_msg = "Data files not loaded"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Log request details
        log_request_details(request)
        
        # Update model parameters from request (only if provided)
        param_updates = {}
        if request.operating_prd is not None:
            model.PARAMS['operating_prd'] = request.operating_prd
            param_updates['operating_prd'] = request.operating_prd
        if request.construction_prd is not None:
            model.PARAMS['construction_prd'] = request.construction_prd
            param_updates['construction_prd'] = request.construction_prd
        if request.util_fac_year1 is not None:
            model.PARAMS['util_fac_year1'] = request.util_fac_year1
            param_updates['util_fac_year1'] = request.util_fac_year1
        if request.util_fac_year2 is not None:
            model.PARAMS['util_fac_year2'] = request.util_fac_year2
            param_updates['util_fac_year2'] = request.util_fac_year2
        if request.util_fac_remaining is not None:
            model.PARAMS['util_fac_remaining'] = request.util_fac_remaining
            param_updates['util_fac_remaining'] = request.util_fac_remaining
        if request.infl is not None:
            model.PARAMS['Infl'] = request.infl
            param_updates['Infl'] = request.infl
        if request.RR is not None:
            model.PARAMS['RR'] = request.RR
            param_updates['RR'] = request.RR
        if request.IRR is not None:
            model.PARAMS['IRR'] = request.IRR
            param_updates['IRR'] = request.IRR
        if request.capex_spread is not None:
            model.PARAMS['capex_spread'] = request.capex_spread
            param_updates['capex_spread'] = request.capex_spread
        if request.shrDebt is not None:
            model.PARAMS['shrDebt'] = request.shrDebt
            param_updates['shrDebt'] = request.shrDebt
        if request.ownerCost is not None:
            model.PARAMS['OwnerCost'] = request.ownerCost
            param_updates['OwnerCost'] = request.ownerCost
        if request.credit is not None:
            model.PARAMS['credit'] = request.credit
            param_updates['credit'] = request.credit
        if request.PRIcoef is not None:
            model.PARAMS['PRIcoef'] = request.PRIcoef
            param_updates['PRIcoef'] = request.PRIcoef
        if request.CONcoef is not None:
            model.PARAMS['CONcoef'] = request.CONcoef
            param_updates['CONcoef'] = request.CONcoef
        if request.EcNatGas is not None:
            model.PARAMS['EcNatGas'] = request.EcNatGas
            param_updates['EcNatGas'] = request.EcNatGas
        if request.ngCcontnt is not None:
            model.PARAMS['ngCcontnt'] = request.ngCcontnt
            param_updates['ngCcontnt'] = request.ngCcontnt
        if request.eEFF is not None:
            model.PARAMS['eEFF'] = request.eEFF
            param_updates['eEFF'] = request.eEFF
        if request.elEFF is not None:
            model.PARAMS['elEFF'] = request.elEFF
            param_updates['elEFF'] = request.elEFF
        if request.hEFF is not None:
            model.PARAMS['hEFF'] = request.hEFF
            param_updates['hEFF'] = request.hEFF

        if param_updates:
            logger.info("Updated model parameters:")
            for param, value in param_updates.items():
                logger.info(f"  {param}: {value}")

        # Filter project data for this request
        project_data = PROJECT_DATA[
            (PROJECT_DATA['Country'] == request.location) & 
            (PROJECT_DATA['Main_Prod'] == request.product)
        ]
        
        if len(project_data) == 0:
            error_msg = f"No project data found for location '{request.location}' and product '{request.product}'"
            logger.error(error_msg)
            logger.error(f"Available locations: {PROJECT_DATA['Country'].unique()}")
            logger.error(f"Available products: {PROJECT_DATA['Main_Prod'].unique()}")
            raise HTTPException(status_code=404, detail=error_msg)

        # Log data details
        log_data_details(project_data)
        
        # Update project data with any provided overrides
        project_updates = {}
        for idx in project_data.index:
            if request.baseYear is not None:
                project_data.at[idx, 'Base_Yr'] = request.baseYear
                project_updates['Base_Yr'] = request.baseYear
            if request.corpTAX is not None:
                project_data.at[idx, 'corpTAX'] = request.corpTAX
                project_updates['corpTAX'] = request.corpTAX
            if request.Feed_Price is not None:
                project_data.at[idx, 'Feed_Price'] = request.Feed_Price
                project_updates['Feed_Price'] = request.Feed_Price
            if request.Fuel_Price is not None:
                project_data.at[idx, 'Fuel_Price'] = request.Fuel_Price
                project_updates['Fuel_Price'] = request.Fuel_Price
            if request.Elect_Price is not None:
                project_data.at[idx, 'Elect_Price'] = request.Elect_Price
                project_updates['Elect_Price'] = request.Elect_Price
            if request.CO2price is not None:
                project_data.at[idx, 'CO2price'] = request.CO2price
                project_updates['CO2price'] = request.CO2price
            if request.CAPEX is not None:
                project_data.at[idx, 'CAPEX'] = request.CAPEX
                project_updates['CAPEX'] = request.CAPEX
            if request.OPEX is not None:
                project_data.at[idx, 'OPEX'] = request.OPEX
                project_updates['OPEX'] = request.OPEX
            if request.Cap is not None:
                project_data.at[idx, 'Cap'] = request.Cap
                project_updates['Cap'] = request.Cap
            if request.Yld is not None:
                project_data.at[idx, 'Yld'] = request.Yld
                project_updates['Yld'] = request.Yld
            if request.feedEcontnt is not None:
                project_data.at[idx, 'feedEcontnt'] = request.feedEcontnt
                project_updates['feedEcontnt'] = request.feedEcontnt
            if request.Heat_req is not None:
                project_data.at[idx, 'Heat_req'] = request.Heat_req
                project_updates['Heat_req'] = request.Heat_req
            if request.Elect_req is not None:
                project_data.at[idx, 'Elect_req'] = request.Elect_req
                project_updates['Elect_req'] = request.Elect_req
            if request.feedCcontnt is not None:
                project_data.at[idx, 'feedCcontnt'] = request.feedCcontnt
                project_updates['feedCcontnt'] = request.feedCcontnt

        if project_updates:
            logger.info("Updated project data fields:")
            for field, value in project_updates.items():
                logger.info(f"  {field}: {value}")

        # Log function call details
        log_function_call_details()

        # Run the analysis with comprehensive logging
        logger.info("=== STARTING MODEL EXECUTION ===")
        
        with capture_model_logs() as (log_capture, stdout_capture, stderr_capture):
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
                
                # Capture all logs from the model execution
                log_contents = log_capture.getvalue()
                stdout_contents = stdout_capture.getvalue()
                stderr_contents = stderr_capture.getvalue()
                
                # Log the captured outputs
                if log_contents:
                    logger.info("=== MODEL LOG OUTPUT ===")
                    for line in log_contents.split('\n'):
                        if line.strip():
                            logger.info(f"MODEL LOG: {line}")
                
                if stdout_contents:
                    logger.info("=== MODEL STDOUT ===")
                    for line in stdout_contents.split('\n'):
                        if line.strip():
                            logger.info(f"MODEL STDOUT: {line}")
                
                if stderr_contents:
                    logger.info("=== MODEL STDERR ===")
                    for line in stderr_contents.split('\n'):
                        if line.strip():
                            logger.error(f"MODEL STDERR: {line}")
                
            except Exception as model_error:
                # Capture any logs that occurred before the exception
                log_contents = log_capture.getvalue()
                stdout_contents = stdout_capture.getvalue()
                stderr_contents = stderr_capture.getvalue()
                
                logger.error("=== MODEL EXECUTION FAILED ===")
                logger.error(f"Model error: {str(model_error)}")
                logger.error(traceback.format_exc())
                
                if log_contents:
                    logger.error("=== MODEL LOGS BEFORE FAILURE ===")
                    for line in log_contents.split('\n'):
                        if line.strip():
                            logger.error(f"MODEL LOG: {line}")
                
                if stderr_contents:
                    logger.error("=== MODEL STDERR BEFORE FAILURE ===")
                    for line in stderr_contents.split('\n'):
                        if line.strip():
                            logger.error(f"MODEL STDERR: {line}")
                
                raise model_error

        logger.info("=== MODEL EXECUTION COMPLETED SUCCESSFULLY ===")
        logger.info(f"Results shape: {results.shape}")
        logger.info(f"Results columns: {list(results.columns)}")
        if len(results) > 0:
            logger.info("Sample result row:")
            sample_row = results.iloc[0]
            for col in results.columns:
                logger.info(f"  {col}: {sample_row[col]}")
        
        # Convert results to list of dicts for JSON response
        return results.to_dict(orient='records')

    except HTTPException:
        logger.error("HTTPException raised during analysis")
        raise
    except Exception as e:
        logger.critical(f"Analysis failed with unexpected error: {str(e)}")
        logger.critical(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        logger.info("=== ANALYSIS REQUEST COMPLETED ===")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_loaded": MULTIPLIER_DATA is not None and PROJECT_DATA is not None,
        "multiplier_data_shape": MULTIPLIER_DATA.shape if MULTIPLIER_DATA is not None else None,
        "project_data_shape": PROJECT_DATA.shape if PROJECT_DATA is not None else None
    }

@app.get("/available_data")
async def get_available_data():
    """Get available locations and products"""
    if PROJECT_DATA is None:
        raise HTTPException(status_code=500, detail="Project data not loaded")
    
    return {
        "locations": PROJECT_DATA['Country'].unique().tolist(),
        "products": PROJECT_DATA['Main_Prod'].unique().tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
