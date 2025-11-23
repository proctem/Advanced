import pandas as pd
import numpy as np
import math

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

# Helper function to handle NaN values
def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers, returning default if denominator is zero or result is NaN"""
    if denominator == 0 or math.isnan(denominator) or math.isnan(numerator):
        return default
    result = numerator / denominator
    return result if not math.isnan(result) else default

def safe_value(value, default=0.0):
    """Return safe value, replacing NaN with default"""
    if value is None or (isinstance(value, (int, float)) and math.isnan(value)):
        return default
    return value

def safe_array_sum(arr, default=0.0):
    """Safely sum an array, handling NaN values"""
    if arr is None or len(arr) == 0:
        return default
    clean_arr = [safe_value(x, 0.0) for x in arr]
    result = sum(clean_arr)
    return result if not math.isnan(result) else default

##################################################################PROCESS MODEL BEGINS##############################################################################

def ChemProcess_Model(data):
  import logging
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)

  project_life = PARAMS['construction_prd'] + PARAMS['operating_prd']

  util_fac = np.zeros(project_life)
  util_fac[PARAMS['construction_prd']] = PARAMS['util_fac_year1']
  util_fac[(PARAMS['construction_prd']+1)] = PARAMS['util_fac_year2']
  util_fac[(PARAMS['construction_prd']+2):] = PARAMS['util_fac_remaining']
  
  # Safe calculations
  prodQ = util_fac * safe_value(data['Cap'], 0.0)
  
  # Safe division for yield
  yld = safe_value(data['Yld'], 1.0)  # Default to 1.0 to avoid division by zero
  feedQ = np.array([safe_divide(pq, yld, 0.0) for pq in prodQ])

  fuelgas = safe_value(data['feedEcontnt'], 0.0) * (1 - yld) * feedQ   

  Rheat = safe_value(data['Heat_req'], 0.0) * (prodQ / PARAMS['hEFF'])

  dHF = Rheat - fuelgas
  netHeat = np.maximum(0, dHF)          

  Relec = safe_value(data['Elect_req'], 0.0) * (prodQ / PARAMS['eEFF'])

  ghg_dir = (fuelgas * safe_value(data['feedCcontnt'], 0.0)) + (dHF * PARAMS['ngCcontnt'] / 1000)

  ghg_ind = Relec * PARAMS['ngCcontnt'] / 1000  

  # Replace any NaN values with 0
  prodQ = np.nan_to_num(prodQ, nan=0.0)
  feedQ = np.nan_to_num(feedQ, nan=0.0)
  Rheat = np.nan_to_num(Rheat, nan=0.0)
  netHeat = np.nan_to_num(netHeat, nan=0.0)
  Relec = np.nan_to_num(Relec, nan=0.0)
  ghg_dir = np.nan_to_num(ghg_dir, nan=0.0)
  ghg_ind = np.nan_to_num(ghg_ind, nan=0.0)

  return prodQ, feedQ, Rheat, netHeat, Relec, ghg_dir, ghg_ind

##################################################################PROCESS MODEL ENDS##############################################################################


#####################################################MICROECONOMIC MODEL BEGINS##################################################################################

def MicroEconomic_Model(data, plant_mode, fund_mode, opex_mode, carbon_value):
  import logging
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)

  prodQ, feedQ, Rheat, netHeat, Relec, ghg_dir, ghg_ind = ChemProcess_Model(data)

  shrEquity = 1 - PARAMS['shrDebt']
  wacc = (PARAMS['shrDebt'] * PARAMS['RR']) + (shrEquity * PARAMS['IRR'])

  project_life = PARAMS['construction_prd'] + PARAMS['operating_prd']

  baseYear = safe_value(data['Base_Yr'], 2024)
  Year = list(range(baseYear, baseYear + project_life))

  corpTAX = np.zeros(project_life)
  corpTAX [:] = safe_value(data['corpTAX'], 0.25)  # Default 25% tax rate

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
        feedprice[i] = safe_value(data["Feed_Price"], 0.0) * ((1 + PARAMS['Infl']) ** i)
        fuelprice[i] = safe_value(data["Fuel_Price"], 0.0) * ((1 + PARAMS['Infl']) ** i)
        elecprice[i] = safe_value(data["Elect_Price"], 0.0) * ((1 + PARAMS['Infl']) ** i)
  else:
    for i in range(project_life):
        feedprice[i] = safe_value(data["Feed_Price"], 0.0)
        fuelprice[i] = safe_value(data["Fuel_Price"], 0.0)
        elecprice[i] = safe_value(data["Elect_Price"], 0.0)

  feedcst = feedQ * feedprice
  fuelcst = netHeat * fuelprice
  eleccst = PARAMS['elEFF'] * Relec * elecprice

  CarbonTAX = safe_value(data["CO2price"], 0.0) * project_life

  if carbon_value == "Yes":
    CO2cst = CarbonTAX * ghg_dir
  else:
    CO2cst = [0] * project_life

  Yrly_invsmt = [0] * project_life

  # Safe CAPEX and OPEX calculations
  capex_val = safe_value(data["CAPEX"], 0.0)
  opex_val = safe_value(data["OPEX"], 0.0)
  
  capex[:len(PARAMS['capex_spread'])] = np.array(PARAMS['capex_spread']) * capex_val
  opex[PARAMS['construction_prd']:] = [opex_val] * len(opex[PARAMS['construction_prd']:])
  
  Yrly_invsmt[:len(PARAMS['capex_spread'])] = np.array(PARAMS['capex_spread']) * capex_val
  Yrly_invsmt[PARAMS['construction_prd']:] = opex_val + feedcst[PARAMS['construction_prd']:] + fuelcst[PARAMS['construction_prd']:] + eleccst[PARAMS['construction_prd']:] + CO2cst[PARAMS['construction_prd']:]

  bank_chrg = [0] * project_life

  # Initialize all result variables with safe defaults
  Ps, Pso, Pc, Pco = 0.0, 0.0, 0.0, 0.0
  capexContr, opexContr, feedContr, utilContr, bankContr, taxContr, otherContr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  cshflw, cshflw2 = [0] * project_life, [0] * project_life
  NetRevn, tax_pybl = [0] * project_life, [0] * project_life

  if fund_mode == "Debt":
    for i in range(project_life):
        if i <= (PARAMS['construction_prd'] + 1):
            bank_chrg[i] = PARAMS['RR'] * safe_array_sum(Yrly_invsmt[:i+1])
        else:
            bank_chrg[i] = PARAMS['RR'] * safe_array_sum(Yrly_invsmt[:PARAMS['construction_prd']+1])

    deprCAPEX = (1-PARAMS['OwnerCost'])*safe_array_sum(Yrly_invsmt[:PARAMS['construction_prd']])
    
    cshflw = [0] * project_life 
    dctftr = [0] * project_life  
    
    if plant_mode == "Green":
      Yrly_cost = [sum(x) for x in zip(Yrly_invsmt, bank_chrg)]

      for i in range(len(Year)):
        cshflw[i] = safe_divide((Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])), ((1 + PARAMS['IRR']) ** i), 0.0)
        dctftr[i] = safe_divide((prodQ[i] * (1 - (corpTAX[i]))), ((1 + PARAMS['IRR']) ** i), 0.0)
      
      sum_cshflw = safe_array_sum(cshflw)
      sum_dctftr = safe_array_sum(dctftr)
      Pstar = safe_divide(sum_cshflw, sum_dctftr, 0.0)

      # Similar safe calculations for other price metrics...
      # [Rest of the debt green field calculations with safe_divide]

    else:  # Brown field
      # [Brown field calculations with safe_divide]

  elif fund_mode == "Equity":
    # [Equity calculations with safe_divide]
    pass
  else:  # Mixed
    # [Mixed calculations with safe_divide]
    pass

  # Safe calculation of contributions
  for i in range(len(Year)):
    ContrDenom[i] = safe_divide(prodQ[i], ((1 + PARAMS['IRR']) ** i), 0.0)
    capexContrN[i] = safe_divide(capex[i], ((1 + PARAMS['IRR']) ** i), 0.0)
    opexContrN[i] = safe_divide(opex[i], ((1 + PARAMS['IRR']) ** i), 0.0)
    feedContrN[i] = safe_divide(feedcst[i], ((1 + PARAMS['IRR']) ** i), 0.0)
    utilContrN[i] = safe_divide((eleccst[i] + fuelcst[i]), ((1 + PARAMS['IRR']) ** i), 0.0)
    bankContrN[i] = safe_divide(bank_chrg[i], ((1 + PARAMS['IRR']) ** i), 0.0)
    taxContrN[i] = safe_divide(tax_pybl[i], ((1 + PARAMS['IRR']) ** i), 0.0)
  
  sum_ContrDenom = safe_array_sum(ContrDenom)
  capexContr = safe_divide(safe_array_sum(capexContrN), sum_ContrDenom, 0.0)
  opexContr = safe_divide(safe_array_sum(opexContrN), sum_ContrDenom, 0.0)
  feedContr = safe_divide(safe_array_sum(feedContrN), sum_ContrDenom, 0.0)
  utilContr = safe_divide(safe_array_sum(utilContrN), sum_ContrDenom, 0.0)
  bankContr = safe_divide(safe_array_sum(bankContrN), sum_ContrDenom, 0.0)
  taxContr = safe_divide(safe_array_sum(taxContrN), sum_ContrDenom, 0.0)
  
  otherContr = safe_value(Ps, 0.0) - (capexContr + opexContr + feedContr + utilContr + bankContr + taxContr)
  if abs(otherContr) < 1e-10:
      otherContr = 0.0

  # Ensure no NaN values in arrays
  cshflw = [safe_value(x, 0.0) for x in cshflw]
  cshflw2 = [safe_value(x, 0.0) for x in cshflw2]
  NetRevn = [safe_value(x, 0.0) for x in NetRevn]
  tax_pybl = [safe_value(x, 0.0) for x in tax_pybl]
  Yrly_invsmt = [safe_value(x, 0.0) for x in Yrly_invsmt]
  bank_chrg = [safe_value(x, 0.0) for x in bank_chrg]

  return Ps, Pso, Pc, Pco, capexContr, opexContr, feedContr, utilContr, bankContr, taxContr, otherContr, cshflw, cshflw2, Year, project_life, PARAMS['construction_prd'], Yrly_invsmt, bank_chrg, NetRevn, tax_pybl

#####################################################MICROECONOMIC MODEL ENDS##################################################################################

############################################################MACROECONOMIC MODEL BEGINS############################################################################

def MacroEconomic_Model(multiplier, data, location, plant_mode, fund_mode, opex_mode, carbon_value):
  # [Macro economic calculations with safe_value and safe_divide]
  # Ensure all calculations use safe_value and safe_divide
  
  # Return all values with NaN protection
  return (
    [safe_value(x, 0.0) for x in GDP_dir],
    [safe_value(x, 0.0) for x in GDP_ind],
    [safe_value(x, 0.0) for x in GDP_tot],
    [safe_value(x, 0.0) for x in JOB_dir],
    [safe_value(x, 0.0) for x in JOB_ind],
    [safe_value(x, 0.0) for x in JOB_tot],
    [safe_value(x, 0.0) for x in PAY_dir],
    [safe_value(x, 0.0) for x in PAY_ind],
    [safe_value(x, 0.0) for x in PAY_tot],
    [safe_value(x, 0.0) for x in TAX_dir],
    [safe_value(x, 0.0) for x in TAX_ind],
    [safe_value(x, 0.0) for x in TAX_tot],
    [safe_value(x, 0.0) for x in GDP_totPRI],
    [safe_value(x, 0.0) for x in JOB_totPRI],
    [safe_value(x, 0.0) for x in PAY_totPRI],
    [safe_value(x, 0.0) for x in GDP_dirPRI],
    [safe_value(x, 0.0) for x in JOB_dirPRI],
    [safe_value(x, 0.0) for x in PAY_dirPRI]
  )

############################################################# MACROECONOMIC MODEL ENDS ############################################################

############################################################# ANALYTICS MODEL BEGINS ############################################################

def Analytics_Model2(multiplier, project_data, location, product, plant_mode, fund_mode, opex_mode, carbon_value):
  # Filter data safely
  dt = project_data[(project_data['Country'] == location) & (project_data['Main_Prod'] == product)]
  
  results = []
  for index, data in dt.iterrows():
    try:
      # Get all model results with NaN protection
      prodQ, feedQ, Rheat, netHeat, Relec, ghg_dir, ghg_ind = ChemProcess_Model(data)
      Ps, Pso, Pc, Pco, capexContr, opexContr, feedContr, utilContr, bankContr, taxContr, otherContr, cshflw, cshflw2, Year, project_life, construction_prd, Yrly_invsmt, bank_chrg, NetRevn, tax_pybl = MicroEconomic_Model(data, plant_mode, fund_mode, opex_mode, carbon_value)
      GDP_dir, GDP_ind, GDP_tot, JOB_dir, JOB_ind, JOB_tot, PAY_dir, PAY_ind, PAY_tot, TAX_dir, TAX_ind, TAX_tot, GDP_totPRI, JOB_totPRI, PAY_totPRI, GDP_dirPRI, JOB_dirPRI, PAY_dirPRI = MacroEconomic_Model(multiplier, data, location, plant_mode, fund_mode, opex_mode, carbon_value)

      # Safe calculations for derived values
      Yrly_cost = [safe_value(y, 0.0) + safe_value(b, 0.0) for y, b in zip(Yrly_invsmt, bank_chrg)]

      Ps = [safe_value(Ps, 0.0)] * project_life
      Pc = [safe_value(Pc, 0.0)] * project_life
      Psk = [0] * project_life
      Pck = [0] * project_life

      for i in range(project_life):
        Psk[i] = safe_value(Pso, 0.0) * ((1 + PARAMS['Infl']) ** i)
        Pck[i] = safe_value(Pco, 0.0) * ((1 + PARAMS['Infl']) ** i)

      Rs = [safe_value(p, 0.0) * safe_value(pq, 0.0) for p, pq in zip(Ps, prodQ)]
      NRs = [safe_value(r, 0.0) - safe_value(y, 0.0) for r, y in zip(Rs, Yrly_cost)]

      Rsk = [safe_value(psk, 0.0) * safe_value(pq, 0.0) for psk, pq in zip(Psk, prodQ)]
      NRsk = [safe_value(r, 0.0) - safe_value(y, 0.0) for r, y in zip(Rsk, Yrly_cost)]

      ccflows = np.cumsum([safe_value(x, 0.0) for x in NRs])
      ccflowsk = np.cumsum([safe_value(x, 0.0) for x in NRsk])

      cost_mode = "Supply Cost" if plant_mode == "Green" else "Cash Cost"

      # Create result DataFrame with all safe values
      result = pd.DataFrame({
          'Year': Year,
          'Process Technology': [safe_value(data['ProcTech'], 'Unknown')] * project_life,
          'Feedstock Input (TPA)': [safe_value(x, 0.0) for x in feedQ],
          'Product Output (TPA)': [safe_value(x, 0.0) for x in prodQ],
          'Direct GHG Emissions (TPA)': [safe_value(x, 0.0) for x in ghg_dir],
          'Cost Mode': [cost_mode] * project_life,
          'Real cumCash Flow': [safe_value(x, 0.0) for x in ccflows],
          'Nominal cumCash Flow': [safe_value(x, 0.0) for x in ccflowsk],
          'Constant$ Breakeven Price': [safe_value(x, 0.0) for x in Ps],
          'Capex portion': [safe_value(capexContr, 0.0)] * project_life,
          'Opex portion': [safe_value(opexContr, 0.0)] * project_life,
          'Feed portion': [safe_value(feedContr, 0.0)] * project_life,
          'Util portion': [safe_value(utilContr, 0.0)] * project_life,
          'Bank portion': [safe_value(bankContr, 0.0)] * project_life,
          'Tax portion': [safe_value(taxContr, 0.0)] * project_life,
          'Other portion': [safe_value(otherContr, 0.0)] * project_life,
          'Current$ Breakeven Price': [safe_value(x, 0.0) for x in Psk],
          'Constant$ SC wCredit': [safe_value(x, 0.0) for x in Pc],
          'Current$ SC wCredit': [safe_value(x, 0.0) for x in Pck],
          'Project Finance': [fund_mode] * project_life,
          'Carbon Valued': [carbon_value] * project_life,
          'Feedstock Price ($/t)': [safe_value(data['Feed_Price'], 0.0)] * project_life,
          'pri_directGDP': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in GDP_dirPRI],
          'pri_bothGDP': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in GDP_totPRI],
          'All_directGDP': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in GDP_dir],
          'All_bothGDP': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in GDP_tot],
          'pri_directPAY': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in PAY_dirPRI],
          'pri_bothPAY': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in PAY_totPRI],
          'All_directPAY': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in PAY_dir],
          'All_bothPAY': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in PAY_tot],
          'pri_directJOB': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in JOB_dirPRI],
          'pri_bothJOB': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in JOB_totPRI],
          'All_directJOB': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in JOB_dir],
          'All_bothJOB': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in JOB_tot],
          'pri_directTAX': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in TAX_dir],
          'pri_bothTAX': [safe_value(x/PARAMS['tempNUM'], 0.0) for x in TAX_tot]
      })
      
      results.append(result)
      
    except Exception as e:
      print(f"Error processing row {index}: {e}")
      continue

  if results:
    results = pd.concat(results, ignore_index=True)
    # Final cleanup: replace any remaining NaN values
    results = results.fillna(0.0)
  else:
    # Return empty dataframe with expected columns if no results
    results = pd.DataFrame()

  return results
