import pandas as pd
import numpy as np
import logging

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

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chemical_plant_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

##################################################################PROCESS MODEL BEGINS##############################################################################

def ChemProcess_Model(data):
    """
    Chemical Process Model for calculating production quantities, energy requirements, and GHG emissions
    """
    try:
        logger.info("Starting Chemical Process Model calculation")
        logger.info(f"Input data: {data.to_dict() if hasattr(data, 'to_dict') else data}")

        project_life = PARAMS['construction_prd'] + PARAMS['operating_prd']
        logger.info(f"Project life: {project_life} years ({PARAMS['construction_prd']} construction + {PARAMS['operating_prd']} operation)")

        # Utility factors calculation
        util_fac = np.zeros(project_life)
        util_fac[PARAMS['construction_prd']] = PARAMS['util_fac_year1']
        util_fac[(PARAMS['construction_prd']+1)] = PARAMS['util_fac_year2']
        util_fac[(PARAMS['construction_prd']+2):] = PARAMS['util_fac_remaining']
        
        logger.info(f"Utility factors: {util_fac}")
        logger.info(f"Capacity: {data['Cap']}")

        # Production quantity calculation
        prodQ = util_fac * data['Cap']
        logger.info(f"Production quantity: {prodQ}")

        # Feedstock quantity calculation
        if data['Yld'] <= 0 or np.isnan(data['Yld']):
            logger.error(f"Invalid yield value: {data['Yld']}")
            raise ValueError(f"Yield must be positive, got: {data['Yld']}")
        
        feedQ = prodQ / data['Yld']
        logger.info(f"Feedstock quantity: {feedQ}")

        # Fuel gas calculation
        fuelgas = data['feedEcontnt'] * (1 - data['Yld']) * feedQ   
        logger.info(f"Fuel gas: {fuelgas}")

        # Heat requirement calculation
        Rheat = data['Heat_req'] * (prodQ / PARAMS['hEFF'])
        logger.info(f"Heat requirement: {Rheat}")

        # Net heat calculation
        dHF = Rheat - fuelgas
        netHeat = np.maximum(0, dHF)          
        logger.info(f"Net heat: {netHeat}")

        # Electricity requirement calculation
        Relec = data['Elect_req'] * (prodQ / PARAMS['eEFF'])
        logger.info(f"Electricity requirement: {Relec}")

        # GHG emissions calculation
        ghg_dir = (fuelgas * data['feedCcontnt']) + (dHF * PARAMS['ngCcontnt'] / 1000)
        ghg_ind = Relec * PARAMS['ngCcontnt'] / 1000  
        
        logger.info(f"Direct GHG emissions: {ghg_dir}")
        logger.info(f"Indirect GHG emissions: {ghg_ind}")

        # Check for NaN values
        if np.any(np.isnan(prodQ)) or np.any(np.isnan(feedQ)) or np.any(np.isnan(ghg_dir)):
            logger.error("NaN values detected in process model outputs")
            raise ValueError("NaN values in process model calculations")

        logger.info("Chemical Process Model completed successfully")
        return prodQ, feedQ, Rheat, netHeat, Relec, ghg_dir, ghg_ind

    except Exception as e:
        logger.error(f"Error in ChemProcess_Model: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

##################################################################PROCESS MODEL ENDS##############################################################################


#####################################################MICROECONOMIC MODEL BEGINS##################################################################################

def MicroEconomic_Model(data, plant_mode, fund_mode, opex_mode, carbon_value):
    """
    Microeconomic Model for calculating financial metrics and costs
    """
    try:
        logger.info("Starting MicroEconomic Model calculation")
        logger.info(f"Plant mode: {plant_mode}, Funding mode: {fund_mode}, OPEX mode: {opex_mode}, Carbon value: {carbon_value}")

        # Get process model outputs
        prodQ, feedQ, Rheat, netHeat, Relec, ghg_dir, ghg_ind = ChemProcess_Model(data)
        
        # Financial parameters
        shrEquity = 1 - PARAMS['shrDebt']
        wacc = (PARAMS['shrDebt'] * PARAMS['RR']) + (shrEquity * PARAMS['IRR'])
        logger.info(f"Financial parameters - Debt share: {PARAMS['shrDebt']}, RR: {PARAMS['RR']}, IRR: {PARAMS['IRR']}, WACC: {wacc}")

        project_life = PARAMS['construction_prd'] + PARAMS['operating_prd']
        baseYear = data['Base_Yr']
        Year = list(range(baseYear, baseYear + project_life))

        # Corporate tax setup
        corpTAX = np.zeros(project_life)
        corpTAX[:] = data['corpTAX']
        corpTAX[:PARAMS['construction_prd']] = 0
        logger.info(f"Corporate tax setup: {corpTAX}")

        # Price arrays initialization
        feedprice = np.zeros(project_life)
        fuelprice = np.zeros(project_life)
        elecprice = np.zeros(project_life)

        # Cost arrays initialization
        capex = np.zeros(project_life)
        opex = np.zeros(project_life)
        capexContrN = np.zeros(project_life)
        opexContrN = np.zeros(project_life)
        feedContrN = np.zeros(project_life)
        utilContrN = np.zeros(project_life)
        bankContrN = np.zeros(project_life)
        taxContrN = np.zeros(project_life)
        ContrDenom = np.zeros(project_life)

        # Price calculation based on inflation mode
        if opex_mode == "Inflated":
            logger.info("Using inflated prices")
            for i in range(project_life):
                feedprice[i] = data["Feed_Price"] * ((1 + PARAMS['Infl']) ** i)
                fuelprice[i] = data["Fuel_Price"] * ((1 + PARAMS['Infl']) ** i)
                elecprice[i] = data["Elect_Price"] * ((1 + PARAMS['Infl']) ** i)
        else:
            logger.info("Using constant prices")
            for i in range(project_life):
                feedprice[i] = data["Feed_Price"]
                fuelprice[i] = data["Fuel_Price"]
                elecprice[i] = data["Elect_Price"]

        logger.info(f"Feed prices (first 5 years): {feedprice[:5]}")
        logger.info(f"Fuel prices (first 5 years): {fuelprice[:5]}")
        logger.info(f"Electricity prices (first 5 years): {elecprice[:5]}")

        # Cost calculations
        feedcst = feedQ * feedprice
        fuelcst = netHeat * fuelprice
        eleccst = PARAMS['elEFF'] * Relec * elecprice

        logger.info(f"Feed cost (first 5 years): {feedcst[:5]}")
        logger.info(f"Fuel cost (first 5 years): {fuelcst[:5]}")
        logger.info(f"Electricity cost (first 5 years): {eleccst[:5]}")

        # Carbon cost calculation
        CarbonTAX = data["CO2price"] * project_life
        if carbon_value == "Yes":
            CO2cst = CarbonTAX * ghg_dir
            logger.info("Carbon cost included in calculations")
        else:
            CO2cst = np.zeros(project_life)
            logger.info("Carbon cost excluded from calculations")

        # Investment arrays
        Yrly_invsmt = np.zeros(project_life)
        capex[:len(PARAMS['capex_spread'])] = np.array(PARAMS['capex_spread']) * data["CAPEX"]
        opex[PARAMS['construction_prd']:] = data["OPEX"]
        
        Yrly_invsmt[:len(PARAMS['capex_spread'])] = np.array(PARAMS['capex_spread']) * data["CAPEX"]
        Yrly_invsmt[PARAMS['construction_prd']:] = (data["OPEX"] + feedcst[PARAMS['construction_prd']:] + 
                                                   fuelcst[PARAMS['construction_prd']:] + 
                                                   eleccst[PARAMS['construction_prd']:] + 
                                                   CO2cst[PARAMS['construction_prd']:])

        logger.info(f"CAPEX distribution: {capex}")
        logger.info(f"OPEX distribution: {opex}")
        logger.info(f"Yearly investment (first 5 years): {Yrly_invsmt[:5]}")

        # Bank charges initialization
        bank_chrg = np.zeros(project_life)

        # Check for NaN values in critical inputs
        if np.any(np.isnan(Yrly_invsmt)) or np.any(np.isnan(prodQ)):
            logger.error("NaN values detected in economic model inputs")
            raise ValueError("NaN values in economic model inputs")

        # Debt financing mode
        if fund_mode == "Debt":
            logger.info("Using Debt financing mode")
            for i in range(project_life):
                if i <= (PARAMS['construction_prd'] + 1):
                    bank_chrg[i] = PARAMS['RR'] * np.sum(Yrly_invsmt[:i+1])
                else:
                    bank_chrg[i] = PARAMS['RR'] * np.sum(Yrly_invsmt[:PARAMS['construction_prd']+1])

            deprCAPEX = (1 - PARAMS['OwnerCost']) * np.sum(Yrly_invsmt[:PARAMS['construction_prd']])
            
            # Green field plant mode
            if plant_mode == "Green":
                logger.info("Green field plant mode")
                Yrly_cost = Yrly_invsmt + bank_chrg
                
                # Initial price calculations
                cshflw = np.zeros(project_life)
                dctftr = np.zeros(project_life)
                
                for i in range(len(Year)):
                    cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - corpTAX[i]) / ((1 + PARAMS['IRR']) ** i)
                    dctftr[i] = (prodQ[i] * (1 - corpTAX[i])) / ((1 + PARAMS['IRR']) ** i)
                
                Pstar = np.sum(cshflw) / np.sum(dctftr)
                
                # Additional calculations for different price scenarios
                for i in range(len(Year)):
                    cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - corpTAX[i]) / ((1 + PARAMS['IRR']) ** i)
                    dctftr[i] = (prodQ[i] * (1 - corpTAX[i]) * ((1 + PARAMS['Infl']) ** i)) / ((1 + PARAMS['IRR']) ** i)
                
                Pstaro = np.sum(cshflw) / np.sum(dctftr)
                Pstark = np.zeros(project_life)
                for i in range(project_life):
                    Pstark[i] = Pstaro * ((1 + PARAMS['Infl']) ** i)
                Rstark = Pstark * prodQ

                NetRevn = Rstark - Yrly_cost

                # Bank charge adjustments
                for i in range(PARAMS['construction_prd'] + 1, project_life):
                    if np.sum(NetRevn[:i]) - np.sum(bank_chrg[:i - 1]) < 0:
                        bank_chrg[i] = PARAMS['RR'] * abs(np.sum(NetRevn[:i]) - np.sum(bank_chrg[:i - 1]))
                    else:
                        bank_chrg[i] = 0

                TIC = data['CAPEX'] + np.sum(bank_chrg)

                # Tax and cash flow calculations
                tax_pybl = np.zeros(project_life)
                depr_asst = 0
                cshflw2 = np.zeros(project_life)
                dctftr2 = np.zeros(project_life)

                for i in range(len(Year)):
                    if NetRevn[i] <= 0:
                        tax_pybl[i] = 0
                        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                        dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                        dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                        cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                    else:
                        if depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) < deprCAPEX:
                            tax_pybl[i] = 0
                            depr_asst += NetRevn[i]
                            cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                            dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                            dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                            cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                        elif depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) > deprCAPEX:
                            tax_pybl[i] = (NetRevn[i] + depr_asst - deprCAPEX) * corpTAX[i]
                            depr_asst += (deprCAPEX - depr_asst)
                            cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + PARAMS['IRR']) ** i)
                            dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                            dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                            cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - PARAMS['credit'])) / ((1 + PARAMS['IRR']) ** i)
                        elif depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) == deprCAPEX:
                            tax_pybl[i] = 0
                            depr_asst += NetRevn[i]
                            cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                            dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                            dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                            cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                        else:
                            tax_pybl[i] = NetRevn[i] * corpTAX[i]
                            cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + PARAMS['IRR']) ** i)
                            dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                            dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                            cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - PARAMS['credit'])) / ((1 + PARAMS['IRR']) ** i)

                Ps = np.sum(cshflw) / np.sum(dctftr)
                Pso = np.sum(cshflw) / np.sum(dctftr2)
                Pc = np.sum(cshflw2) / np.sum(dctftr)
                Pco = np.sum(cshflw2) / np.sum(dctftr2)

            else:  # Brown field plant mode
                logger.info("Brown field plant mode")
                # Similar calculations for brown field (simplified for brevity)
                bank_chrg = np.zeros(project_life)
                Yrly_invsmt[:PARAMS['construction_prd']] = 0
                Yrly_cost = Yrly_invsmt + bank_chrg

                # Price calculations for brown field
                cshflw = np.zeros(project_life)
                dctftr = np.zeros(project_life)
                
                for i in range(len(Year)):
                    cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - corpTAX[i]) / ((1 + PARAMS['IRR']) ** i)
                    dctftr[i] = (prodQ[i] * (1 - corpTAX[i])) / ((1 + PARAMS['IRR']) ** i)
                
                Pstar = np.sum(cshflw) / np.sum(dctftr)
                
                # Additional price scenarios
                for i in range(len(Year)):
                    cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - corpTAX[i]) / ((1 + PARAMS['IRR']) ** i)
                    dctftr[i] = (prodQ[i] * (1 - corpTAX[i]) * ((1 + PARAMS['Infl']) ** i)) / ((1 + PARAMS['IRR']) ** i)
                
                Pstaro = np.sum(cshflw) / np.sum(dctftr)
                Pstark = np.zeros(project_life)
                for i in range(project_life):
                    Pstark[i] = Pstaro * ((1 + PARAMS['Infl']) ** i)
                Rstark = Pstark * prodQ

                NetRevn = Rstark - Yrly_cost

                # Bank charge adjustments for brown field
                for i in range(PARAMS['construction_prd'] + 1, project_life):
                    if np.sum(NetRevn[:i]) - np.sum(bank_chrg[:i - 1]) < 0:
                        bank_chrg[i] = PARAMS['RR'] * abs(np.sum(NetRevn[:i]) - np.sum(bank_chrg[:i - 1]))
                    else:
                        bank_chrg[i] = 0

                TIC = data['CAPEX'] + np.sum(bank_chrg)

                # Tax calculations for brown field
                tax_pybl = np.zeros(project_life)
                cshflw2 = np.zeros(project_life)
                dctftr2 = np.zeros(project_life)

                for i in range(len(Year)):
                    if NetRevn[i] <= 0:
                        tax_pybl[i] = 0
                        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                        dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                        dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                        cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                    else:
                        tax_pybl[i] = NetRevn[i] * corpTAX[i]
                        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + PARAMS['IRR']) ** i)
                        dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                        dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                        cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - PARAMS['credit'])) / ((1 + PARAMS['IRR']) ** i)

                Ps = np.sum(cshflw) / np.sum(dctftr)
                Pso = np.sum(cshflw) / np.sum(dctftr2)
                Pc = np.sum(cshflw2) / np.sum(dctftr)
                Pco = np.sum(cshflw2) / np.sum(dctftr2)

        elif fund_mode == "Equity":
            logger.info("Using Equity financing mode")
            # Equity financing calculations (similar structure to debt but with different parameters)
            bank_chrg = np.zeros(project_life)
            deprCAPEX = (1 - PARAMS['OwnerCost']) * np.sum(Yrly_invsmt[:PARAMS['construction_prd']])
            
            if plant_mode == "Green":
                logger.info("Green field plant mode with equity financing")
                # Similar calculations as debt but with equity parameters
                Yrly_cost = Yrly_invsmt + bank_chrg
                
                cshflw = np.zeros(project_life)
                dctftr = np.zeros(project_life)
                
                for i in range(len(Year)):
                    cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - corpTAX[i]) / ((1 + PARAMS['IRR']) ** i)
                    dctftr[i] = (prodQ[i] * (1 - corpTAX[i])) / ((1 + PARAMS['IRR']) ** i)
                
                Pstar = np.sum(cshflw) / np.sum(dctftr)
                
                # Additional calculations
                for i in range(len(Year)):
                    cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - corpTAX[i]) / ((1 + PARAMS['IRR']) ** i)
                    dctftr[i] = (prodQ[i] * (1 - corpTAX[i]) * ((1 + PARAMS['Infl']) ** i)) / ((1 + PARAMS['IRR']) ** i)
                
                Pstaro = np.sum(cshflw) / np.sum(dctftr)
                Pstark = np.zeros(project_life)
                for i in range(project_life):
                    Pstark[i] = Pstaro * ((1 + PARAMS['Infl']) ** i)
                Rstark = Pstark * prodQ

                NetRevn = Rstark - Yrly_cost
                TIC = data['CAPEX'] + np.sum(bank_chrg)

                # Tax calculations
                tax_pybl = np.zeros(project_life)
                depr_asst = 0
                cshflw2 = np.zeros(project_life)
                dctftr2 = np.zeros(project_life)

                for i in range(len(Year)):
                    if NetRevn[i] <= 0:
                        tax_pybl[i] = 0
                        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                        dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                        dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                        cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                    else:
                        if depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) < deprCAPEX:
                            tax_pybl[i] = 0
                            depr_asst += NetRevn[i]
                            cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                            dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                            dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                            cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                        elif depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) > deprCAPEX:
                            tax_pybl[i] = (NetRevn[i] + depr_asst - deprCAPEX) * corpTAX[i]
                            depr_asst += (deprCAPEX - depr_asst)
                            cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + PARAMS['IRR']) ** i)
                            dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                            dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                            cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - PARAMS['credit'])) / ((1 + PARAMS['IRR']) ** i)
                        elif depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) == deprCAPEX:
                            tax_pybl[i] = 0
                            depr_asst += NetRevn[i]
                            cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                            dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                            dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                            cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                        else:
                            tax_pybl[i] = NetRevn[i] * corpTAX[i]
                            cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + PARAMS['IRR']) ** i)
                            dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                            dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                            cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - PARAMS['credit'])) / ((1 + PARAMS['IRR']) ** i)

                Ps = np.sum(cshflw) / np.sum(dctftr)
                Pso = np.sum(cshflw) / np.sum(dctftr2)
                Pc = np.sum(cshflw2) / np.sum(dctftr)
                Pco = np.sum(cshflw2) / np.sum(dctftr2)

            else:  # Brown field with equity
                logger.info("Brown field plant mode with equity financing")
                bank_chrg = np.zeros(project_life)
                Yrly_invsmt[:PARAMS['construction_prd']] = 0
                Yrly_cost = Yrly_invsmt + bank_chrg

                cshflw = np.zeros(project_life)
                dctftr = np.zeros(project_life)
                
                for i in range(len(Year)):
                    cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - corpTAX[i]) / ((1 + PARAMS['IRR']) ** i)
                    dctftr[i] = (prodQ[i] * (1 - corpTAX[i])) / ((1 + PARAMS['IRR']) ** i)
                
                Pstar = np.sum(cshflw) / np.sum(dctftr)
                
                for i in range(len(Year)):
                    cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - corpTAX[i]) / ((1 + PARAMS['IRR']) ** i)
                    dctftr[i] = (prodQ[i] * (1 - corpTAX[i]) * ((1 + PARAMS['Infl']) ** i)) / ((1 + PARAMS['IRR']) ** i)
                
                Pstaro = np.sum(cshflw) / np.sum(dctftr)
                Pstark = np.zeros(project_life)
                for i in range(project_life):
                    Pstark[i] = Pstaro * ((1 + PARAMS['Infl']) ** i)
                Rstark = Pstark * prodQ

                NetRevn = Rstark - Yrly_cost
                TIC = data['CAPEX'] + np.sum(bank_chrg)

                tax_pybl = np.zeros(project_life)
                cshflw2 = np.zeros(project_life)
                dctftr2 = np.zeros(project_life)

                for i in range(len(Year)):
                    if NetRevn[i] <= 0:
                        tax_pybl[i] = 0
                        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                        dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                        dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                        cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + PARAMS['IRR']) ** i)
                    else:
                        tax_pybl[i] = NetRevn[i] * corpTAX[i]
                        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + PARAMS['IRR']) ** i)
                        dctftr[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
                        dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + PARAMS['IRR']) ** i)
                        cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - PARAMS['credit'])) / ((1 + PARAMS['IRR']) ** i)

                Ps = np.sum(cshflw) / np.sum(dctftr)
                Pso = np.sum(cshflw) / np.sum(dctftr2)
                Pc = np.sum(cshflw2) / np.sum(dctftr)
                Pco = np.sum(cshflw2) / np.sum(dctftr2)

        else:  # Mixed financing mode
            logger.info("Using Mixed financing mode")
            for i in range(project_life):
                if i <= (PARAMS['construction_prd'] + 1):
                    bank_chrg[i] = PARAMS['RR'] * PARAMS['shrDebt'] * np.sum(Yrly_invsmt[:i+1])
                else:
                    bank_chrg[i] = PARAMS['RR'] * PARAMS['shrDebt'] * np.sum(Yrly_invsmt[:PARAMS['construction_prd']+1])

            deprCAPEX = (1 - PARAMS['OwnerCost']) * np.sum(Yrly_invsmt[:PARAMS['construction_prd']])
            
            if plant_mode == "Green":
                logger.info("Green field plant mode with mixed financing")
                Yrly_cost = Yrly_invsmt + bank_chrg
                
                cshflw = np.zeros(project_life)
                dctftr = np.zeros(project_life)
                
                for i in range(len(Year)):
                    cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - corpTAX[i]) / ((1 + wacc) ** i)
                    dctftr[i] = (prodQ[i] * (1 - corpTAX[i])) / ((1 + wacc) ** i)
                
                Pstar = np.sum(cshflw) / np.sum(dctftr)
                
                for i in range(len(Year)):
                    cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - corpTAX[i]) / ((1 + wacc) ** i)
                    dctftr[i] = (prodQ[i] * (1 - corpTAX[i]) * ((1 + PARAMS['Infl']) ** i)) / ((1 + wacc) ** i)
                
                Pstaro = np.sum(cshflw) / np.sum(dctftr)
                Pstark = np.zeros(project_life)
                for i in range(project_life):
                    Pstark[i] = Pstaro * ((1 + PARAMS['Infl']) ** i)
                Rstark = Pstark * prodQ

                NetRevn = Rstark - Yrly_cost

                for i in range(PARAMS['construction_prd'] + 1, project_life):
                    if np.sum(NetRevn[:i]) - np.sum(bank_chrg[:i - 1]) < 0:
                        bank_chrg[i] = PARAMS['RR'] * abs(np.sum(NetRevn[:i]) - np.sum(bank_chrg[:i - 1]))
                    else:
                        bank_chrg[i] = 0

                TIC = data['CAPEX'] + np.sum(bank_chrg)

                tax_pybl = np.zeros(project_life)
                depr_asst = 0
                cshflw2 = np.zeros(project_life)
                dctftr2 = np.zeros(project_life)

                for i in range(len(Year)):
                    if NetRevn[i] <= 0:
                        tax_pybl[i] = 0
                        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
                        dctftr[i] = prodQ[i] / ((1 + wacc) ** i)
                        dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + wacc) ** i)
                        cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
                    else:
                        if depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) < deprCAPEX:
                            tax_pybl[i] = 0
                            depr_asst += NetRevn[i]
                            cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
                            dctftr[i] = prodQ[i] / ((1 + wacc) ** i)
                            dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + wacc) ** i)
                            cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
                        elif depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) > deprCAPEX:
                            tax_pybl[i] = (NetRevn[i] + depr_asst - deprCAPEX) * corpTAX[i]
                            depr_asst += (deprCAPEX - depr_asst)
                            cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + wacc) ** i)
                            dctftr[i] = prodQ[i] / ((1 + wacc) ** i)
                            dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + wacc) ** i)
                            cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - PARAMS['credit'])) / ((1 + wacc) ** i)
                        elif depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) == deprCAPEX:
                            tax_pybl[i] = 0
                            depr_asst += NetRevn[i]
                            cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
                            dctftr[i] = prodQ[i] / ((1 + wacc) ** i)
                            dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + wacc) ** i)
                            cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
                        else:
                            tax_pybl[i] = NetRevn[i] * corpTAX[i]
                            cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + wacc) ** i)
                            dctftr[i] = prodQ[i] / ((1 + wacc) ** i)
                            dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + wacc) ** i)
                            cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - PARAMS['credit'])) / ((1 + wacc) ** i)

                Ps = np.sum(cshflw) / np.sum(dctftr)
                Pso = np.sum(cshflw) / np.sum(dctftr2)
                Pc = np.sum(cshflw2) / np.sum(dctftr)
                Pco = np.sum(cshflw2) / np.sum(dctftr2)

            else:  # Brown field with mixed financing
                logger.info("Brown field plant mode with mixed financing")
                bank_chrg = np.zeros(project_life)
                Yrly_invsmt[:PARAMS['construction_prd']] = 0
                Yrly_cost = Yrly_invsmt + bank_chrg

                cshflw = np.zeros(project_life)
                dctftr = np.zeros(project_life)
                
                for i in range(len(Year)):
                    cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - corpTAX[i]) / ((1 + wacc) ** i)
                    dctftr[i] = (prodQ[i] * (1 - corpTAX[i])) / ((1 + wacc) ** i)
                
                Pstar = np.sum(cshflw) / np.sum(dctftr)
                
                for i in range(len(Year)):
                    cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - corpTAX[i]) / ((1 + wacc) ** i)
                    dctftr[i] = (prodQ[i] * (1 - corpTAX[i]) * ((1 + PARAMS['Infl']) ** i)) / ((1 + wacc) ** i)
                
                Pstaro = np.sum(cshflw) / np.sum(dctftr)
                Pstark = np.zeros(project_life)
                for i in range(project_life):
                    Pstark[i] = Pstaro * ((1 + PARAMS['Infl']) ** i)
                Rstark = Pstark * prodQ

                NetRevn = Rstark - Yrly_cost
                TIC = data['CAPEX'] + np.sum(bank_chrg)

                tax_pybl = np.zeros(project_life)
                cshflw2 = np.zeros(project_life)
                dctftr2 = np.zeros(project_life)

                for i in range(len(Year)):
                    if NetRevn[i] <= 0:
                        tax_pybl[i] = 0
                        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
                        dctftr[i] = prodQ[i] / ((1 + wacc) ** i)
                        dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + wacc) ** i)
                        cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
                    else:
                        tax_pybl[i] = NetRevn[i] * corpTAX[i]
                        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + wacc) ** i)
                        dctftr[i] = prodQ[i] / ((1 + wacc) ** i)
                        dctftr2[i] = prodQ[i] * ((1 + PARAMS['Infl']) ** i) / ((1 + wacc) ** i)
                        cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - PARAMS['credit'])) / ((1 + wacc) ** i)

                Ps = np.sum(cshflw) / np.sum(dctftr)
                Pso = np.sum(cshflw) / np.sum(dctftr2)
                Pc = np.sum(cshflw2) / np.sum(dctftr)
                Pco = np.sum(cshflw2) / np.sum(dctftr2)

        # Contribution calculations
        logger.info("Calculating cost contributions")
        for i in range(len(Year)):
            ContrDenom[i] = prodQ[i] / ((1 + PARAMS['IRR']) ** i)
            capexContrN[i] = capex[i] / ((1 + PARAMS['IRR']) ** i)
            opexContrN[i] = opex[i] / ((1 + PARAMS['IRR']) ** i)
            feedContrN[i] = feedcst[i] / ((1 + PARAMS['IRR']) ** i)
            utilContrN[i] = (eleccst[i] + fuelcst[i]) / ((1 + PARAMS['IRR']) ** i)
            bankContrN[i] = bank_chrg[i] / ((1 + PARAMS['IRR']) ** i)
            taxContrN[i] = tax_pybl[i] / ((1 + PARAMS['IRR']) ** i)

        capexContr = np.sum(capexContrN) / np.sum(ContrDenom)
        opexContr = np.sum(opexContrN) / np.sum(ContrDenom)
        feedContr = np.sum(feedContrN) / np.sum(ContrDenom)
        utilContr = np.sum(utilContrN) / np.sum(ContrDenom)
        bankContr = np.sum(bankContrN) / np.sum(ContrDenom)
        taxContr = np.sum(taxContrN) / np.sum(ContrDenom)
        
        otherContr = round(Ps - (capexContr + opexContr + feedContr + utilContr + bankContr + taxContr), 10)
        if abs(otherContr) < 1e-10:
            otherContr = 0.0

        # Check for NaN values in outputs
        outputs = [Ps, Pso, Pc, Pco, capexContr, opexContr, feedContr, utilContr, bankContr, taxContr, otherContr]
        output_names = ['Ps', 'Pso', 'Pc', 'Pco', 'capexContr', 'opexContr', 'feedContr', 'utilContr', 'bankContr', 'taxContr', 'otherContr']
        
        for name, value in zip(output_names, outputs):
            if np.isnan(value) or np.isinf(value):
                logger.error(f"Invalid value in {name}: {value}")
                raise ValueError(f"Invalid value in {name}: {value}")

        logger.info("MicroEconomic Model completed successfully")
        logger.info(f"Results - Ps: {Ps:.4f}, Pso: {Pso:.4f}, Pc: {Pc:.4f}, Pco: {Pco:.4f}")

        return (Ps, Pso, Pc, Pco, capexContr, opexContr, feedContr, utilContr, bankContr, taxContr, 
                otherContr, cshflw, cshflw2, Year, project_life, PARAMS['construction_prd'], 
                Yrly_invsmt, bank_chrg, NetRevn, tax_pybl)

    except Exception as e:
        logger.error(f"Error in MicroEconomic_Model: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Input data: {data.to_dict() if hasattr(data, 'to_dict') else data}")
        raise

#####################################################MICROECONOMIC MODEL ENDS##################################################################################


############################################################MACROECONOMIC MODEL BEGINS############################################################################

def MacroEconomic_Model(multiplier, data, location, plant_mode, fund_mode, opex_mode, carbon_value):
    """
    Macroeconomic Model for calculating economic impacts
    """
    try:
        logger.info("Starting MacroEconomic Model calculation")
        logger.info(f"Location: {location}, Plant mode: {plant_mode}")

        # Get process and microeconomic model outputs
        prodQ, _, _, _, _, _, _ = ChemProcess_Model(data)
        Ps, _, _, _, _, _, _, _, _, _, _, _, _, Year, project_life, const_prd, Yrly_invsmt, bank_chrg, _, _ = MicroEconomic_Model(data, plant_mode, fund_mode, opex_mode, carbon_value)

        # Investment calculations
        pri_invsmt = np.zeros(project_life)
        con_invsmt = np.zeros(project_life)
        bank_invsmt = np.zeros(project_life)

        pri_invsmt[:const_prd] = [PARAMS['PRIcoef'] * Yrly_invsmt[i] for i in range(const_prd)]
        pri_invsmt[const_prd:] = data["OPEX"]
        con_invsmt[:const_prd] = [PARAMS['CONcoef'] * Yrly_invsmt[i] for i in range(const_prd)]
        bank_invsmt = bank_chrg

        logger.info(f"Primary investment (first 5 years): {pri_invsmt[:5]}")
        logger.info(f"Construction investment (first 5 years): {con_invsmt[:5]}")
        logger.info(f"Bank investment (first 5 years): {bank_invsmt[:5]}")

        # Multiplier data extraction with error handling
        try:
            # Chemicals and Chemical Products [C20] Multipliers
            output_PRI = multiplier[(multiplier['Country'] == location) &
                                  (multiplier['Multiplier Type'] == "Output Multiplier") &
                                  (multiplier['Sector'] == (location + "_" + "C20"))]

            pay_PRI = multiplier[(multiplier['Country'] == location) &
                               (multiplier['Multiplier Type'] == "Compensation (USD per million USD output)") &
                               (multiplier['Sector'] == (location + "_" + "C20"))]

            job_PRI = multiplier[(multiplier['Country'] == location) &
                               (multiplier['Multiplier Type'] == "Employment Elasticity (Jobs per million USD output)") &
                               (multiplier['Sector'] == (location + "_" + "C20"))]

            tax_PRI = multiplier[(multiplier['Country'] == location) &
                               (multiplier['Multiplier Type'] == "Tax Revenue Share (USD per million USD output)") &
                               (multiplier['Sector'] == (location + "_" + "C20"))]

            gdp_PRI = multiplier[(multiplier['Country'] == location) &
                               (multiplier['Multiplier Type'] == "Value-Added Share (USD per million USD output)") &
                               (multiplier['Sector'] == (location + "_" + "C20"))]

            # Check if multiplier data is available
            if (output_PRI.empty or pay_PRI.empty or job_PRI.empty or 
                tax_PRI.empty or gdp_PRI.empty):
                logger.warning(f"Incomplete multiplier data for {location}_C20, using default values")
                # Use default values or raise exception based on requirements
                raise ValueError(f"Incomplete multiplier data for {location}_C20")

        except Exception as e:
            logger.error(f"Error extracting multiplier data: {str(e)}")
            raise

        # Convert to pandas Series for vectorized operations
        pri_invsmt_series = pd.Series(pri_invsmt)
        con_invsmt_series = pd.Series(con_invsmt)
        bank_invsmt_series = pd.Series(bank_invsmt)

        # GDP impacts calculation
        try:
            GDP_dirPRI = gdp_PRI['Direct Impact'].values[0] * pri_invsmt_series
            GDP_totPRI = gdp_PRI['Total Impact'].values[0] * pri_invsmt_series
            GDP_dir = GDP_dirPRI  # Simplified for this example
            GDP_tot = GDP_totPRI  # Simplified for this example
            
            logger.info(f"GDP direct impacts (first 5 years): {GDP_dirPRI[:5]}")
            logger.info(f"GDP total impacts (first 5 years): {GDP_totPRI[:5]}")

        except Exception as e:
            logger.error(f"Error in GDP impact calculation: {str(e)}")
            raise

        # Job impacts calculation
        try:
            JOB_dirPRI = job_PRI['Direct Impact'].values[0] * pri_invsmt_series
            JOB_totPRI = job_PRI['Total Impact'].values[0] * pri_invsmt_series
            JOB_dir = JOB_dirPRI  # Simplified for this example
            JOB_tot = JOB_totPRI  # Simplified for this example
            
            logger.info(f"Job direct impacts (first 5 years): {JOB_dirPRI[:5]}")
            logger.info(f"Job total impacts (first 5 years): {JOB_totPRI[:5]}")

        except Exception as e:
            logger.error(f"Error in job impact calculation: {str(e)}")
            raise

        # Payment impacts calculation
        try:
            PAY_dirPRI = pay_PRI['Direct Impact'].values[0] * pri_invsmt_series
            PAY_totPRI = pay_PRI['Total Impact'].values[0] * pri_invsmt_series
            PAY_dir = PAY_dirPRI  # Simplified for this example
            PAY_tot = PAY_totPRI  # Simplified for this example
            
            logger.info(f"Payment direct impacts (first 5 years): {PAY_dirPRI[:5]}")
            logger.info(f"Payment total impacts (first 5 years): {PAY_totPRI[:5]}")

        except Exception as e:
            logger.error(f"Error in payment impact calculation: {str(e)}")
            raise

        # Tax impacts calculation
        try:
            TAX_dir = np.zeros(project_life)
            TAX_ind = np.zeros(project_life)
            TAX_tot = np.zeros(project_life)

            for i in range(const_prd, project_life):
                TAX_dir[i] = tax_PRI['Direct Impact'].values[0] * np.array(Yrly_invsmt[i] + (Ps * prodQ[i]))
                TAX_ind[i] = tax_PRI['Indirect Impact'].values[0] * np.array(Yrly_invsmt[i] + (Ps * prodQ[i]))
                TAX_tot[i] = tax_PRI['Total Impact'].values[0] * np.array(Yrly_invsmt[i] + (Ps * prodQ[i]))
            
            logger.info(f"Tax impacts calculated for operating period")

        except Exception as e:
            logger.error(f"Error in tax impact calculation: {str(e)}")
            raise

        # Check for NaN values in outputs
        outputs_to_check = [GDP_dir, GDP_tot, JOB_dir, JOB_tot, PAY_dir, PAY_tot, TAX_dir, TAX_tot]
        output_names = ['GDP_dir', 'GDP_tot', 'JOB_dir', 'JOB_tot', 'PAY_dir', 'PAY_tot', 'TAX_dir', 'TAX_tot']
        
        for name, output in zip(output_names, outputs_to_check):
            if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                logger.error(f"NaN or Inf values detected in {name}")
                raise ValueError(f"Invalid values in {name}")

        logger.info("MacroEconomic Model completed successfully")
        return (GDP_dir, GDP_tot, GDP_tot, JOB_dir, JOB_tot, JOB_tot, PAY_dir, PAY_tot, PAY_tot, 
                TAX_dir, TAX_tot, TAX_tot, GDP_totPRI, JOB_totPRI, PAY_totPRI, GDP_dirPRI, JOB_dirPRI, PAY_dirPRI)

    except Exception as e:
        logger.error(f"Error in MacroEconomic_Model: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

############################################################# MACROECONOMIC MODEL ENDS ############################################################


############################################################# ANALYTICS MODEL BEGINS ############################################################

def Analytics_Model2(multiplier, project_data, location, product, plant_mode, fund_mode, opex_mode, carbon_value):
    """
    Analytics Model for comprehensive analysis and results compilation
    """
    try:
        logger.info("Starting Analytics Model")
        logger.info(f"Parameters - Location: {location}, Product: {product}, Plant mode: {plant_mode}, "
                   f"Funding mode: {fund_mode}, OPEX mode: {opex_mode}, Carbon value: {carbon_value}")

        # Filter project data
        dt = project_data[(project_data['Country'] == location) & (project_data['Main_Prod'] == product)]
        
        if dt.empty:
            logger.error(f"No data found for location: {location} and product: {product}")
            raise ValueError(f"No project data available for {location} and {product}")

        logger.info(f"Found {len(dt)} records for analysis")

        results = []
        for index, data in dt.iterrows():
            try:
                logger.info(f"Processing record {index + 1}/{len(dt)}")
                
                # Run all models
                prodQ, feedQ, Rheat, netHeat, Relec, ghg_dir, ghg_ind = ChemProcess_Model(data)
                (Ps, Pso, Pc, Pco, capexContr, opexContr, feedContr, utilContr, bankContr, taxContr, 
                 otherContr, cshflw, cshflw2, Year, project_life, const_prd, Yrly_invsmt, bank_chrg, 
                 NetRevn, tax_pybl) = MicroEconomic_Model(data, plant_mode, fund_mode, opex_mode, carbon_value)
                
                (GDP_dir, GDP_ind, GDP_tot, JOB_dir, JOB_ind, JOB_tot, PAY_dir, PAY_ind, PAY_tot, 
                 TAX_dir, TAX_ind, TAX_tot, GDP_totPRI, JOB_totPRI, PAY_totPRI, GDP_dirPRI, 
                 JOB_dirPRI, PAY_dirPRI) = MacroEconomic_Model(multiplier, data, location, plant_mode, fund_mode, opex_mode, carbon_value)

                # Additional calculations
                Yrly_cost = np.array(Yrly_invsmt) + np.array(bank_chrg)

                Ps_array = np.full(project_life, Ps)
                Pc_array = np.full(project_life, Pc)
                Psk = np.zeros(project_life)
                Pck = np.zeros(project_life)

                for i in range(project_life):
                    Psk[i] = Pso * ((1 + PARAMS['Infl']) ** i)
                    Pck[i] = Pco * ((1 + PARAMS['Infl']) ** i)

                Rs = Ps_array * prodQ
                NRs = Rs - Yrly_cost

                Rsk = Psk * prodQ
                NRsk = Rsk - Yrly_cost

                ccflows = np.cumsum(NRs)
                ccflowsk = np.cumsum(NRsk)

                cost_mode = "Supply Cost" if plant_mode == "Green" else "Cash Cost"

                # Create result dataframe
                result = pd.DataFrame({
                    'Year': Year,
                    'Process Technology': [data['ProcTech']] * project_life,
                    'Plant Size': [data['Plant_Size']] * project_life,
                    'Plant Efficiency': [data['Plant_Effy']] * project_life,
                    'Feedstock Input (TPA)': feedQ,
                    'Product Output (TPA)': prodQ,
                    'Direct GHG Emissions (TPA)': ghg_dir,
                    'Cost Mode': [cost_mode] * project_life,
                    'Real cumCash Flow': ccflows,
                    'Nominal cumCash Flow': ccflowsk,
                    'Constant$ Breakeven Price': Ps_array,
                    'Capex portion': [capexContr] * project_life,
                    'Opex portion': [opexContr] * project_life,
                    'Feed portion': [feedContr] * project_life,
                    'Util portion': [utilContr] * project_life,
                    'Bank portion': [bankContr] * project_life,
                    'Tax portion': [taxContr] * project_life,
                    'Other portion': [otherContr] * project_life,
                    'Current$ Breakeven Price': Psk,
                    'Constant$ SC wCredit': Pc_array,
                    'Current$ SC wCredit': Pck,
                    'Project Finance': [fund_mode] * project_life,
                    'Carbon Valued': [carbon_value] * project_life,
                    'Feedstock Price ($/t)': [data['Feed_Price']] * project_life,
                    'pri_directGDP': np.array(GDP_dirPRI) / PARAMS['tempNUM'],
                    'pri_bothGDP': np.array(GDP_totPRI) / PARAMS['tempNUM'],
                    'All_directGDP': np.array(GDP_dir) / PARAMS['tempNUM'],
                    'All_bothGDP': np.array(GDP_tot) / PARAMS['tempNUM'],
                    'pri_directPAY': np.array(PAY_dirPRI) / PARAMS['tempNUM'],
                    'pri_bothPAY': np.array(PAY_totPRI) / PARAMS['tempNUM'],
                    'All_directPAY': np.array(PAY_dir) / PARAMS['tempNUM'],
                    'All_bothPAY': np.array(PAY_tot) / PARAMS['tempNUM'],
                    'pri_directJOB': np.array(JOB_dirPRI) / PARAMS['tempNUM'],
                    'pri_bothJOB': np.array(JOB_totPRI) / PARAMS['tempNUM'],
                    'All_directJOB': np.array(JOB_dir) / PARAMS['tempNUM'],
                    'All_bothJOB': np.array(JOB_tot) / PARAMS['tempNUM'],
                    'pri_directTAX': np.array(TAX_dir) / PARAMS['tempNUM'],
                    'pri_bothTAX': np.array(TAX_tot) / PARAMS['tempNUM']
                })

                # Check for NaN values in final results
                if result.isnull().any().any():
                    nan_columns = result.columns[result.isnull().any()].tolist()
                    logger.error(f"NaN values found in result columns: {nan_columns}")
                    raise ValueError(f"NaN values in result columns: {nan_columns}")

                results.append(result)
                logger.info(f"Successfully processed record {index + 1}")

            except Exception as e:
                logger.error(f"Error processing record {index + 1}: {str(e)}")
                continue

        if not results:
            logger.error("No successful results generated")
            raise ValueError("No results generated from any records")

        # Combine all results
        final_results = pd.concat(results, ignore_index=True)
        
        # Final validation
        if final_results.empty:
            logger.error("Final results dataframe is empty")
            raise ValueError("Final results dataframe is empty")

        logger.info(f"Analytics Model completed successfully. Generated {len(final_results)} records")
        return final_results

    except Exception as e:
        logger.error(f"Error in Analytics_Model2: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

# Additional utility function for data validation
def validate_input_data(data):
    """
    Validate input data for NaN values and data types
    """
    logger.info("Validating input data")
    
    required_fields = ['Cap', 'Yld', 'feedEcontnt', 'Heat_req', 'Elect_req', 'feedCcontnt', 
                      'Base_Yr', 'corpTAX', 'Feed_Price', 'Fuel_Price', 'Elect_Price', 
                      'CO2price', 'CAPEX', 'OPEX']
    
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            raise ValueError(f"Missing required field: {field}")
        
        value = data[field]
        if np.isnan(value) or np.isinf(value):
            logger.error(f"Invalid value in field {field}: {value}")
            raise ValueError(f"Invalid value in field {field}: {value}")
    
    logger.info("Input data validation completed successfully")
    return True

# Example usage with error handling
def run_complete_analysis(multiplier, project_data, location, product, plant_mode, fund_mode, opex_mode, carbon_value):
    """
    Complete analysis runner with comprehensive error handling
    """
    try:
        logger.info("Starting complete analysis run")
        
        # Validate inputs
        if not validate_input_data(project_data.iloc[0] if hasattr(project_data, 'iloc') else project_data):
            raise ValueError("Input data validation failed")
        
        # Run analytics model
        results = Analytics_Model2(multiplier, project_data, location, product, plant_mode, fund_mode, opex_mode, carbon_value)
        
        logger.info("Complete analysis run finished successfully")
        return results
        
    except Exception as e:
        logger.error(f"Complete analysis run failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

############################################################# ANALYTICS MODEL ENDS ############################################################
