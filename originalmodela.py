import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

###############################PARAMSS#################################

# Dictionary for hardcoded parameters
PARAMS = {
    'EcNatGas': 53.6,
    'ngCcontnt': 50.3,
    'hEFF': 0.80,
    'eEFF': 0.50,
    'construction_prd': 3,
    'operating_prd': 27,
    'util_fac_year1': 0.70,
    'util_fac_year2': 0.80,
    'util_fac_remaining': 0.95,
    'elEFF': 0.90,
    'Infl': 0.02,
    'RR': 0.035,
    'IRR': 0.10,
    'shrDebt': 0.60,
    'capex_spread': [0.2,0.5,0.3],
    'OwnerCost': 0.10,
    'credit': 0.10,
    'PRIcoef': 0.9,
    'CONcoef': 0.1,
    'tempNUM': 1000000
}


##################################################################PROCESS MODEL BEGINS##############################################################################

def ChemProcess_Model(data):
    logger.info("Starting ChemProcess_Model")
    logger.info(f"Input data: {data}")

    project_life = PARAMS['construction_prd'] + PARAMS['operating_prd']

    util_fac = np.zeros(project_life)
    util_fac[PARAMS['construction_prd']] = PARAMS['util_fac_year1']
    util_fac[(PARAMS['construction_prd']+1)] = PARAMS['util_fac_year2']
    util_fac[(PARAMS['construction_prd']+2):] = PARAMS['util_fac_remaining']
    
    # Check for NaN in util_fac
    if np.any(np.isnan(util_fac)):
        logger.error(f"NaN found in util_fac: {util_fac}")
        raise ValueError("NaN in utility factors")

    prodQ = util_fac * data['Cap']
    logger.info(f"Product Qty: {prodQ}")
    
    # Check for NaN in prodQ
    if np.any(np.isnan(prodQ)):
        logger.error(f"NaN found in prodQ: {prodQ}")
        raise ValueError("NaN in product quantity")

    feedQ = prodQ / data['Yld']
    
    # Check for NaN in feedQ
    if np.any(np.isnan(feedQ)):
        logger.error(f"NaN found in feedQ: {feedQ}, prodQ: {prodQ}, Yld: {data['Yld']}")
        raise ValueError("NaN in feedstock quantity")

    fuelgas = data['feedEcontnt'] * (1 - data['Yld']) * feedQ   
    
    # Check for NaN in fuelgas
    if np.any(np.isnan(fuelgas)):
        logger.error(f"NaN found in fuelgas: {fuelgas}")
        raise ValueError("NaN in fuelgas")

    Rheat = data['Heat_req'] * (prodQ / PARAMS['hEFF'])
    
    # Check for NaN in Rheat
    if np.any(np.isnan(Rheat)):
        logger.error(f"NaN found in Rheat: {Rheat}")
        raise ValueError("NaN in required heat")

    dHF = Rheat - fuelgas
    netHeat = np.maximum(0, dHF)          
    
    # Check for NaN in netHeat
    if np.any(np.isnan(netHeat)):
        logger.error(f"NaN found in netHeat: {netHeat}, Rheat: {Rheat}, fuelgas: {fuelgas}")
        raise ValueError("NaN in net heat")

    Relec = data['Elect_req'] * (prodQ / PARAMS['eEFF'])
    
    # Check for NaN in Relec
    if np.any(np.isnan(Relec)):
        logger.error(f"NaN found in Relec: {Relec}")
        raise ValueError("NaN in required electricity")

    ghg_dir = (fuelgas * data['feedCcontnt']) + (dHF * PARAMS['ngCcontnt'] / 1000)
    
    # Check for NaN in ghg_dir
    if np.any(np.isnan(ghg_dir)):
        logger.error(f"NaN found in ghg_dir: {ghg_dir}")
        raise ValueError("NaN in direct GHG")

    ghg_ind = Relec * PARAMS['ngCcontnt'] / 1000  
    
    # Check for NaN in ghg_ind
    if np.any(np.isnan(ghg_ind)):
        logger.error(f"NaN found in ghg_ind: {ghg_ind}")
        raise ValueError("NaN in indirect GHG")

    logger.info("ChemProcess_Model completed successfully")
    return prodQ, feedQ, Rheat, netHeat, Relec, ghg_dir, ghg_ind

##################################################################PROCESS MODEL ENDS##############################################################################


#####################################################MICROECONOMIC MODEL BEGINS##################################################################################

def MicroEconomic_Model(data, plant_mode, fund_mode, opex_mode, carbon_value):
    logger.info("Starting MicroEconomic_Model")
    logger.info(f"Inputs - plant_mode: {plant_mode}, fund_mode: {fund_mode}, opex_mode: {opex_mode}, carbon_value: {carbon_value}")

    prodQ, feedQ, Rheat, netHeat, Relec, ghg_dir, ghg_ind = ChemProcess_Model(data)

    shrEquity = 1 - PARAMS['shrDebt']
    wacc = (PARAMS['shrDebt'] * PARAMS['RR']) + (shrEquity * PARAMS['IRR'])
    logger.info(f"WACC calculated: {wacc}")

    project_life = PARAMS['construction_prd'] + PARAMS['operating_prd']
    baseYear = data['Base_Yr']
    Year = list(range(baseYear, baseYear + project_life))

    corpTAX = np.zeros(project_life)
    corpTAX[:] = data['corpTAX']
    corpTAX[:PARAMS['construction_prd']] = 0

    feedprice = [0] * project_life
    fuelprice = [0] * project_life
    elecprice = [0] * project_life

    capex = [0] * project_life
    opex = [0] * project_life
    capexContrN = [0] * project_life
    opexContrN = [0] * project_life
    feedContrN = [0] * project_life
    utilContrN = [0] * project_life
    bankContrN = [0] * project_life
    taxContrN = [0] * project_life
    ContrDenom = [0] * project_life

    if opex_mode == "Inflated":
        for i in range(project_life):
            feedprice[i] = data["Feed_Price"] * ((1 + PARAMS['Infl']) ** i)
            fuelprice[i] = data["Fuel_Price"] * ((1 + PARAMS['Infl']) ** i)
            elecprice[i] = data["Elect_Price"] * ((1 + PARAMS['Infl']) ** i)
    else:
        for i in range(project_life):
            feedprice[i] = data["Feed_Price"]
            fuelprice[i] = data["Fuel_Price"]
            elecprice[i] = data["Elect_Price"]

    feedcst = feedQ * feedprice
    fuelcst = netHeat * fuelprice
    eleccst = PARAMS['elEFF'] * Relec * elecprice

    CarbonTAX = data["CO2price"] * project_life

    if carbon_value == "Yes":
        CO2cst = CarbonTAX * ghg_dir
    else:
        CO2cst = [0] * project_life

    Yrly_invsmt = [0] * project_life

    capex[:len(PARAMS['capex_spread'])] = np.array(PARAMS['capex_spread']) * data["CAPEX"]
    opex[PARAMS['construction_prd']:] = [data["OPEX"]] * len(opex[PARAMS['construction_prd']:])
    
    Yrly_invsmt[:len(PARAMS['capex_spread'])] = np.array(PARAMS['capex_spread']) * data["CAPEX"]
    Yrly_invsmt[PARAMS['construction_prd']:] = data["OPEX"] + feedcst[PARAMS['construction_prd']:] + fuelcst[PARAMS['construction_prd']:] + eleccst[PARAMS['construction_prd']:] + CO2cst[PARAMS['construction_prd']:]

    # Check for NaN in key arrays
    if np.any(np.isnan(feedcst)):
        logger.error(f"NaN found in feedcst: {feedcst}")
        raise ValueError("NaN in feed cost")
    if np.any(np.isnan(fuelcst)):
        logger.error(f"NaN found in fuelcst: {fuelcst}")
        raise ValueError("NaN in fuel cost")
    if np.any(np.isnan(eleccst)):
        logger.error(f"NaN found in eleccst: {eleccst}")
        raise ValueError("NaN in electricity cost")
    if np.any(np.isnan(Yrly_invsmt)):
        logger.error(f"NaN found in Yrly_invsmt: {Yrly_invsmt}")
        raise ValueError("NaN in yearly investment")

    bank_chrg = [0] * project_life

    # [Rest of the MicroEconomic_Model code remains the same, but add NaN checks in critical sections]
    
    # Add NaN checks after key calculations
    def check_for_nan(arr, name):
        if np.any(np.isnan(arr)):
            logger.error(f"NaN found in {name}: {arr}")
            raise ValueError(f"NaN in {name}")

    # After major calculations, add checks:
    check_for_nan(np.array(bank_chrg), "bank_chrg")
    check_for_nan(np.array(NetRevn), "NetRevn")
    check_for_nan(np.array(tax_pybl), "tax_pybl")
    check_for_nan(np.array(cshflw), "cshflw")
    check_for_nan(np.array(cshflw2), "cshflw2")
    
    # Check final outputs
    if np.isnan(Ps) or np.isinf(Ps):
        logger.error(f"Invalid Ps value: {Ps}")
        raise ValueError("Invalid Ps value")
    if np.isnan(Pso) or np.isinf(Pso):
        logger.error(f"Invalid Pso value: {Pso}")
        raise ValueError("Invalid Pso value")
    if np.isnan(Pc) or np.isinf(Pc):
        logger.error(f"Invalid Pc value: {Pc}")
        raise ValueError("Invalid Pc value")
    if np.isnan(Pco) or np.isinf(Pco):
        logger.error(f"Invalid Pco value: {Pco}")
        raise ValueError("Invalid Pco value")

    logger.info("MicroEconomic_Model completed successfully")
    return Ps, Pso, Pc, Pco, capexContr, opexContr, feedContr, utilContr, bankContr, taxContr, otherContr, cshflw, cshflw2, Year, project_life, PARAMS['construction_prd'], Yrly_invsmt, bank_chrg, NetRevn, tax_pybl

#####################################################MICROECONOMIC MODEL ENDS##################################################################################


############################################################MACROECONOMIC MODEL BEGINS############################################################################

def MacroEconomic_Model(multiplier, data, location, plant_mode, fund_mode, opex_mode, carbon_value):
    logger.info("Starting MacroEconomic_Model")

    prodQ, _, _, _, _, _, _ = ChemProcess_Model(data)
    Ps, _, _, _, _, _, _, _, _, _, _, _, _, Year, project_life, PARAMS['construction_prd'], Yrly_invsmt, bank_chrg, _, _ = MicroEconomic_Model(data, plant_mode, fund_mode, opex_mode, carbon_value)

    # [Rest of MacroEconomic_Model code remains the same, but add NaN checks]
    
    # Check final outputs for NaN
    def check_econ_outputs(arr, name):
        if np.any(np.isnan(arr)):
            logger.error(f"NaN found in {name}: {arr}")
            raise ValueError(f"NaN in {name}")

    GDP_dir, GDP_ind, GDP_tot, JOB_dir, JOB_ind, JOB_tot, PAY_dir, PAY_ind, PAY_tot, TAX_dir, TAX_ind, TAX_tot, GDP_totPRI, JOB_totPRI, PAY_totPRI, GDP_dirPRI, JOB_dirPRI, PAY_dirPRI = [None] * 18  # Placeholder

    # [Your existing MacroEconomic_Model calculations here]
    
    # After calculations, check all outputs
    outputs = [GDP_dir, GDP_ind, GDP_tot, JOB_dir, JOB_ind, JOB_tot, PAY_dir, PAY_ind, PAY_tot, TAX_dir, TAX_ind, TAX_tot, GDP_totPRI, JOB_totPRI, PAY_totPRI, GDP_dirPRI, JOB_dirPRI, PAY_dirPRI]
    output_names = ['GDP_dir', 'GDP_ind', 'GDP_tot', 'JOB_dir', 'JOB_ind', 'JOB_tot', 'PAY_dir', 'PAY_ind', 'PAY_tot', 'TAX_dir', 'TAX_ind', 'TAX_tot', 'GDP_totPRI', 'JOB_totPRI', 'PAY_totPRI', 'GDP_dirPRI', 'JOB_dirPRI', 'PAY_dirPRI']
    
    for arr, name in zip(outputs, output_names):
        if arr is not None:
            check_econ_outputs(np.array(arr), name)

    logger.info("MacroEconomic_Model completed successfully")
    return GDP_dir, GDP_ind, GDP_tot, JOB_dir, JOB_ind, JOB_tot, PAY_dir, PAY_ind, PAY_tot, TAX_dir, TAX_ind, TAX_tot, GDP_totPRI, JOB_totPRI, PAY_totPRI, GDP_dirPRI, JOB_dirPRI, PAY_dirPRI

############################################################# MACROECONOMIC MODEL ENDS ############################################################


############################################################# ANALYTICS MODEL BEGINS ############################################################

def Analytics_Model2(multiplier, project_data, location, product, plant_mode, fund_mode, opex_mode, carbon_value):
    logger.info("Starting Analytics_Model2")
    logger.info(f"Inputs - location: {location}, product: {product}, plant_mode: {plant_mode}, fund_mode: {fund_mode}, opex_mode: {opex_mode}, carbon_value: {carbon_value}")

    # Filtering data
    dt = project_data[(project_data['Country'] == location) & (project_data['Main_Prod'] == product)]
    logger.info(f"Found {len(dt)} matching records")
    
    if len(dt) == 0:
        logger.error(f"No data found for location: {location}, product: {product}")
        raise ValueError(f"No data found for the specified criteria")

    results = []
    for index, data in dt.iterrows():
        logger.info(f"Processing record {index}")
        
        try:
            prodQ, feedQ, Rheat, netHeat, Relec, ghg_dir, ghg_ind = ChemProcess_Model(data)
            Ps, Pso, Pc, Pco, capexContr, opexContr, feedContr, utilContr, bankContr, taxContr, otherContr, cshflw, cshflw2, Year, project_life, PARAMS['construction_prd'], Yrly_invsmt, bank_chrg, NetRevn, tax_pybl = MicroEconomic_Model(data, plant_mode, fund_mode, opex_mode, carbon_value)
            GDP_dir, GDP_ind, GDP_tot, JOB_dir, JOB_ind, JOB_tot, PAY_dir, PAY_ind, PAY_tot, TAX_dir, TAX_ind, TAX_tot, GDP_totPRI, JOB_totPRI, PAY_totPRI, GDP_dirPRI, JOB_dirPRI, PAY_dirPRI = MacroEconomic_Model(multiplier, data, location, plant_mode, fund_mode, opex_mode, carbon_value)

            # Convert all arrays to lists and replace NaN with None for JSON serialization
            def sanitize_for_json(arr):
                if isinstance(arr, (np.ndarray, list)):
                    return [None if (isinstance(x, (int, float)) and (np.isnan(x) or np.isinf(x))) else x for x in arr]
                elif isinstance(arr, (int, float)) and (np.isnan(arr) or np.isinf(arr)):
                    return None
                return arr

            Yrly_cost = np.array(Yrly_invsmt) + np.array(bank_chrg)

            Ps_list = [Ps] * project_life
            Pc_list = [Pc] * project_life
            Psk = [0] * project_life
            Pck = [0] * project_life

            for i in range(project_life):
                Psk[i] = Pso * ((1 + PARAMS['Infl']) ** i)
                Pck[i] = Pco * ((1 + PARAMS['Infl']) ** i)

            Rs = [Ps_list[i] * prodQ[i] for i in range(project_life)]
            NRs = [Rs[i] - Yrly_cost[i] for i in range(project_life)]

            Rsk = [Psk[i] * prodQ[i] for i in range(project_life)]
            NRsk = [Rsk[i] - Yrly_cost[i] for i in range(project_life)]

            ccflows = np.cumsum(NRs)
            ccflowsk = np.cumsum(NRsk)

            cost_modes = ["Supply Cost", "Cash Cost"]
            cost_mode = cost_modes[0] if plant_mode == "Green" else cost_modes[1]

            # [Rest of your Analytics_Model2 calculations...]

            # Sanitize all arrays before creating DataFrame
            result = pd.DataFrame({
                'Year': Year,
                'Process Technology': [data['ProcTech']] * project_life,
                'Feedstock Input (TPA)': sanitize_for_json(feedQ),
                'Product Output (TPA)': sanitize_for_json(prodQ),
                'Direct GHG Emissions (TPA)': sanitize_for_json(ghg_dir),
                'Cost Mode': [cost_mode] * project_life,
                'Real cumCash Flow': sanitize_for_json(ccflows),
                'Nominal cumCash Flow': sanitize_for_json(ccflowsk),
                'Constant$ Breakeven Price': sanitize_for_json(Ps_list),
                'Capex portion': [sanitize_for_json(capexContr)] * project_life,
                'Opex portion': [sanitize_for_json(opexContr)] * project_life,
                'Feed portion': [sanitize_for_json(feedContr)] * project_life,
                'Util portion': [sanitize_for_json(utilContr)] * project_life,
                'Bank portion': [sanitize_for_json(bankContr)] * project_life,
                'Tax portion': [sanitize_for_json(taxContr)] * project_life,
                'Other portion': [sanitize_for_json(otherContr)] * project_life,
                'Current$ Breakeven Price': sanitize_for_json(Psk),
                'Constant$ SC wCredit': sanitize_for_json(Pc_list),
                'Current$ SC wCredit': sanitize_for_json(Pck),
                'Project Finance': [fund_mode] * project_life,
                'Carbon Valued': [carbon_value] * project_life,
                'Feedstock Price ($/t)': [data['Feed_Price']] * project_life,
                'pri_directGDP': sanitize_for_json(np.array(pri_directGDP)/PARAMS['tempNUM']),
                'pri_bothGDP': sanitize_for_json(np.array(pri_bothGDP)/PARAMS['tempNUM']),
                'All_directGDP': sanitize_for_json(np.array(All_directGDP)/PARAMS['tempNUM']),
                'All_bothGDP': sanitize_for_json(np.array(All_bothGDP)/PARAMS['tempNUM']),
                'pri_directPAY': sanitize_for_json(np.array(pri_directPAY)/PARAMS['tempNUM']),
                'pri_bothPAY': sanitize_for_json(np.array(pri_bothPAY)/PARAMS['tempNUM']),
                'All_directPAY': sanitize_for_json(np.array(All_directPAY)/PARAMS['tempNUM']),
                'All_bothPAY': sanitize_for_json(np.array(All_bothPAY)/PARAMS['tempNUM']),
                'pri_directJOB': sanitize_for_json(np.array(pri_directJOB)/PARAMS['tempNUM']),
                'pri_bothJOB': sanitize_for_json(np.array(pri_bothJOB)/PARAMS['tempNUM']),
                'All_directJOB': sanitize_for_json(np.array(All_directJOB)/PARAMS['tempNUM']),
                'All_bothJOB': sanitize_for_json(np.array(All_bothJOB)/PARAMS['tempNUM']),
                'pri_directTAX': sanitize_for_json(np.array(pri_directTAX)/PARAMS['tempNUM']),
                'pri_bothTAX': sanitize_for_json(np.array(pri_bothTAX)/PARAMS['tempNUM'])
            })
            
            # Check final DataFrame for NaN values
            if result.isnull().values.any():
                nan_columns = result.columns[result.isnull().any()].tolist()
                logger.warning(f"NaN values found in columns: {nan_columns}")
                # Replace remaining NaN with 0 for JSON serialization
                result = result.fillna(0)
                
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing record {index}: {str(e)}")
            logger.error(f"Problematic data: {data}")
            raise

    if results:
        results_df = pd.concat(results, ignore_index=True)
        logger.info("Analytics_Model2 completed successfully")
        return results_df
    else:
        logger.error("No results generated")
        raise ValueError("No results generated from the analysis")
