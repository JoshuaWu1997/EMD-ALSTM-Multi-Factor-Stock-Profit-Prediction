def get_infomation(cat):
    factor_list, select_list, back = None, None, None
    if cat == '0':
        factor_list = ['NetWorkingCapital', 'NetDebt', 'RetainedEarnings', 'GrossProfit', 'FCFF', 'TotalPaidinCapital',
                       'IntFreeNCL', 'IntFreeCL', 'EBIAT', 'EBIT', 'EBITDA', 'NIAPCut', 'WorkingCapital', 'IntDebt',
                       'IntCL', 'NRProfitLoss', 'FCFE', 'TotalFixedAssets', 'ValueChgProfit', 'OperateNetIncome', 'DA',
                       'NetIntExpense', 'NetTangibleAssets', 'TEAP', 'ASSI', 'TotalAssets', 'NIAP', 'COperAdelpct',
                       'COperAdelQpct', 'COperApct', 'COperATTMpct', 'COperDdelpct', 'COperDdelQpct', 'COperDpct',
                       'COperDTTMpct', 'COnonperDdelpct', 'COnonperDdelQpct', 'COnonperDpct', 'COnonperDTTMpct',
                       'COnonperAdelpct', 'COnonperAdelQpct', 'COnonperApct', 'COnonperATTMpct']
        select_list = ['NetWorkingCapital', 'NetDebt', 'RetainedEarnings', 'GrossProfit', 'FCFF', 'TotalPaidinCapital',
                       'IntFreeNCL', 'IntFreeCL', 'EBIT', 'EBITDA', 'NIAPCut', 'WorkingCapital', 'IntDebt', 'IntCL',
                       'NRProfitLoss', 'FCFE', 'TotalFixedAssets', 'ValueChgProfit', 'OperateNetIncome', 'DA',
                       'NetIntExpense', 'NetTangibleAssets', 'TEAP', 'ASSI', 'TotalAssets', 'NIAP', 'COperAdelpct',
                       'COnonperDdelpct', 'COnonperDdelQpct', 'COnonperDTTMpct']
        back = 1

    if cat == '1':
        factor_list = ['DebtEquityRatio', 'SuperQuickRatio', 'NonCurrentAssetsRatio', 'EquityToAsset',
                       'EquityFixedAssetRatio', 'FixAssetRatio', 'CurrentRatio', 'CurrentAssetsRatio', 'QuickRatio',
                       'IntangibleAssetRatio', 'BondsPayableToAsset', 'DebtsAssetRatio', 'LongDebtToWorkingCapital',
                       'LongTermDebtToAsset', 'LongDebtToAsset', 'BLEV', 'DebtTangibleEquityRatio',
                       'CashToCurrentLiability', 'OperCashInToCurrentLiability', 'CurrentAssetsTRate',
                       'AccountsPayablesTRate', 'ROA', 'NOCFToTLiability', 'OperCashInToAsset', 'MLEV',
                       'TSEPToTotalCapital', 'TotalAssetsTRate', 'EquityTRate', 'FinancialExpenseRate',
                       'TotalProfitCostRatio', 'AdminiExpenseRate', 'NPToTOR', 'SalesCostRatio', 'NetProfitRatio',
                       'GrossIncomeRatio', 'TaxRatio', 'OperatingExpenseRate', 'OperatingProfitRatio',
                       'OperatingProfitToTOR', 'EBITToTOR', 'NetNonOIToTP', 'ROAEBITTTM', 'ROE', 'InventoryTRate',
                       'FixedAssetsTRate', 'NOCFToOperatingNI', 'CashRateOfSales', 'SaleServiceCashToOR',
                       'CashRateOfSalesLatest', 'NetNonOIToTPLatest', 'PeriodCostsRate', 'InvestRAssociatesToTP',
                       'InvestRAssociatesToTPLatest', 'DividendCover', 'OperatingNIToTPLatest', 'NPCutToNP',
                       'OperatingNIToTP', 'DividendPaidRatio', 'RetainedEarningRatio', 'DEGM', 'ACCA', 'CFO2EV',
                       'NOCFToOperatingNILatest', 'NOCFToNetDebt', 'NetProfitCashCover', 'InventoryTDays',
                       'OperatingCycle', 'AccountsPayablesTDays', 'ARTRate', 'ARTDays', 'CashConversionCycle',
                       'InteBearDebtToTotalCapital', 'TangibleAToInteBearDebt', 'TangibleAToNetDebt',
                       'TSEPToInterestBearDebt', 'NOCFToInterestBearDebt', 'InterestCover', 'ROIC', 'ROEDiluted',
                       'ROEAvg', 'ROECut', 'ROECutWeighted', 'ROEWeighted', 'ROAEBIT', 'ROE5', 'ROA5']
        select_list = ['DebtEquityRatio', 'SuperQuickRatio', 'NonCurrentAssetsRatio', 'EquityToAsset',
                       'EquityFixedAssetRatio', 'CurrentRatio', 'BondsPayableToAsset', 'LongTermDebtToAsset',
                       'LongDebtToAsset', 'DebtTangibleEquityRatio', 'CashToCurrentLiability',
                       'OperCashInToCurrentLiability', 'CurrentAssetsTRate', 'AccountsPayablesTRate', 'ROA',
                       'TSEPToTotalCapital', 'TotalAssetsTRate', 'EquityTRate', 'NPToTOR', 'SalesCostRatio',
                       'NetProfitRatio', 'GrossIncomeRatio', 'TaxRatio', 'OperatingProfitRatio', 'OperatingProfitToTOR',
                       'EBITToTOR', 'NetNonOIToTP', 'ROAEBITTTM', 'ROE', 'InventoryTRate', 'FixedAssetsTRate',
                       'NOCFToOperatingNI', 'CashRateOfSales', 'PeriodCostsRate', 'InvestRAssociatesToTP',
                       'InvestRAssociatesToTPLatest', 'DividendCover', 'OperatingNIToTPLatest', 'NPCutToNP',
                       'OperatingNIToTP', 'DividendPaidRatio', 'RetainedEarningRatio', 'DEGM', 'ACCA', 'CFO2EV',
                       'NOCFToOperatingNILatest', 'NOCFToNetDebt', 'NetProfitCashCover', 'InventoryTDays',
                       'OperatingCycle', 'AccountsPayablesTDays', 'ARTRate', 'ARTDays', 'CashConversionCycle',
                       'TangibleAToInteBearDebt', 'TangibleAToNetDebt', 'TSEPToInterestBearDebt',
                       'NOCFToInterestBearDebt', 'InterestCover', 'ROIC', 'ROEDiluted', 'ROEAvg', 'ROECut',
                       'ROECutWeighted', 'ROEWeighted', 'ROAEBIT', ]
        back = 1

    if cat == '2':
        factor_list = ['Variance120', 'Variance20', 'Variance60', 'Kurtosis120', 'Kurtosis20', 'Kurtosis60',
                       'Skewness20', 'HBETA', 'HSIGMA', 'DDNBT', 'DDNCR', 'GainVariance20', 'GainVariance60',
                       'GainVariance120', 'LossVariance20', 'LossVariance60', 'LossVariance120',
                       'GainLossVarianceRatio20', 'GainLossVarianceRatio60', 'GainLossVarianceRatio120', 'CMRA12',
                       'CMRA24', 'HsigmaCNE5', 'DASTD', 'DDNSR', 'TOBT', 'DDNSR', 'BackwardADJ', 'Treynorratio20',
                       'Treynorratio120', 'Treynorratio60', 'Sharperatio20', 'Sharperatio60', 'Sharperatio120',
                       'InformationRatio20', 'InformationRatio60', 'InformationRatio120', 'Beta20', 'Beta60', 'Beta120',
                       'Beta252', 'CAPMAlpha20', 'CAPMAlpha60', 'CAPMAlpha120']
        select_list = ['Variance20', 'Kurtosis20', 'Kurtosis60', 'Skewness20', 'HBETA', 'DDNBT', 'DDNCR',
                       'GainVariance20', 'LossVariance20', 'CMRA12', 'CMRA24', 'BackwardADJ', 'Treynorratio20',
                       'Treynorratio60', 'Sharperatio20', 'Sharperatio60', 'InformationRatio20', 'InformationRatio60',
                       'Beta20', 'CAPMAlpha20', 'CAPMAlpha60']
        back = 15

    if cat == '3':
        factor_list = ['VSTD10', 'VOL10', 'TVSTD20', 'TVMA20', 'VSTD20', 'VOL20', 'VOL5', 'VOL60', 'TVSTD6', 'TVMA6',
                       'Volumn1M', 'Volumn3M', 'NVI', 'PVI', 'DAVOL5', 'DAVOL20', 'VOL240', 'DAVOL10', 'VOL120',
                       'MoneyFlow20', 'VROC6', 'VROC12', 'ATR14', 'ATR6', 'VR', 'ACD20', 'ACD6', 'VOSC', 'Volatility',
                       'AR', 'ARBR', 'BR', 'WVAD', 'MAWVAD', 'PSY', 'OBV', 'OBV20', 'OBV6', 'RSI', 'VMACD', 'VDIFF',
                       'VDEA', 'VEMA10', 'VEMA12', 'VEMA26', 'VEMA5', 'JDQS20', 'KlingerOscillator', 'ADTM', 'SBM',
                       'STM', 'STOM', 'STOQ', 'STOA', 'CGO_5', 'CGO_10', 'CGO_60', 'CGO_100', 'CGO_120', 'ST_5',
                       'ST_10', 'ST_20', 'ST_60', 'ST_120', 'TK_20', 'TK_60', 'TK_120', 'cm_ARC', 'cm_VRC', 'cm_SRC',
                       'cm_KRC', 'FR', 'FR_pure']
        select_list = ['VSTD10', 'VOL10', 'VSTD20', 'VROC6', 'AR', 'WVAD', 'VMACD', 'VDIFF', 'VDEA', 'VEMA10', 'STOQ']
        back = 15

    if cat == '4':
        factor_list = ['FinancingCashGrowRate', 'NPParentCompanyGrowRate', 'OperCashGrowRate', 'NetProfitGrowRate',
                       'NetCashFlowGrowRate', 'NetAssetGrowRate', 'TotalProfitGrowRate', 'InvestCashGrowRate',
                       'OperatingProfitGrowRate', 'OperatingRevenueGrowRate', 'TotalAssetGrowRate',
                       'NetProfitGrowRate3Y', 'NetProfitGrowRate5Y', 'OperatingRevenueGrowRate3Y',
                       'OperatingRevenueGrowRate5Y', 'EGRO', 'SGRO']
        select_list = ['FinancingCashGrowRate', 'NPParentCompanyGrowRate', 'OperCashGrowRate', 'NetProfitGrowRate',
                       'NetCashFlowGrowRate', 'NetAssetGrowRate', 'TotalProfitGrowRate', 'InvestCashGrowRate',
                       'OperatingProfitGrowRate', 'OperatingRevenueGrowRate', 'TotalAssetGrowRate']
        back = 1

    if cat == '5':
        factor_list = ['MA10', 'EMA10', 'MA120', 'EMA120', 'EMA12', 'MA20', 'EMA20', 'EMA26', 'TEMA10', 'TEMA5', 'MA5',
                       'EMA5', 'MA60', 'EMA60', 'DHILO', 'MFI', 'BollDown', 'BollUp', 'DBCD', 'MTM', 'MTMMA', 'CR20',
                       'MassIndex', 'SwingIndex', 'ChaikinOscillator', 'UOS', 'plusDI', 'minusDI', 'ADX', 'MACD',
                       'ADXR', 'KDJ_D', 'KDJ_J', 'KDJ_K', 'ILLIQUIDITY', 'ChaikinVolatility', 'Ulcer10', 'Ulcer5',
                       'Elder', 'BBI', 'EMV14', 'EMV6']
        select_list = ['MA10', 'DHILO', 'MFI', 'CR20', 'UOS', 'ADX', 'ADXR', 'KDJ_D', 'KDJ_J', 'ILLIQUIDITY', 'EMV6']
        back = 15

    if cat == '6':
        factor_list = ['REVS250', 'REVS60', 'REVS750', 'REVS10', 'REVS20', 'REVS120', 'REVS5', 'BIAS20', 'BIAS10',
                       'BIAS5', 'BIAS60', 'REVS5M20', 'REVS5m60', 'CCI5', 'CCI10', 'CCI20', 'CCI88', 'SRMI', 'CMO',
                       'CMOSD', 'CMOSU', 'Price1M', 'Price1Y', 'Price3M', 'Rank1M', 'BBIC', 'PVT', 'PVT12', 'PVT6',
                       'APBMA', 'MA10Close', 'PLRC12', 'PLRC6', 'MA10RegressCoeff12', 'MA10RegressCoeff6', 'BULLPOWER',
                       'BEARPOWER', 'PEHIST120', 'PEHIST60', 'PEHIST20', 'PEHIST250', 'AroonUp', 'AroonDown', 'DEA',
                       'DIFF', 'AD', 'AD20', 'AD6', 'ARC', 'FiftyTwoWeekHigh', 'COPPOCKCURVE', 'Aroon', 'RC12', 'RC20',
                       'RC24', 'ROC6', 'DDI', 'DIZ', 'DIF', 'RSTR504', 'RSTR21', 'RSTR42', 'RSTR63', 'RSTR126',
                       'RSTR252', 'RSTR756']
        select_list = ['CCI5', 'BULLPOWER', 'PEHIST60', 'AD', 'RSTR21', 'RSTR42', 'RSTR63']
        back = 15

    if cat == '7':
        factor_list = ['NegMktValue', 'PE', 'PB', 'PS', 'MktValue', 'PCF', 'LFLO', 'LCAP', 'NLSIZE', 'ForwardPE',
                       'StaticPE', 'ETOP', 'CETOP', 'PEG3Y', 'PEG5Y', 'CTOP', 'TA2EV', 'ETP5', 'CTP5']
        select_list = ['NegMktValue', 'PE', 'MktValue', 'StaticPE', 'CTOP', 'TA2EV']
        back = 1

    if cat == '8':
        factor_list = ['BasicEPS', 'EPS', 'DilutedEPS', 'NetAssetPS', 'TORPS', 'OperatingRevenuePS',
                       'OperatingProfitPS', 'EBITPS', 'CapitalSurplusFundPS', 'SurplusReserveFundPS',
                       'UndividedProfitPS', 'RetainedEarningsPS', 'OperCashFlowPS', 'CashFlowPS', 'ShareholderFCFPS',
                       'EnterpriseFCFPS', 'DividendPS']
        select_list = ['BasicEPS', 'NetAssetPS', 'TORPS', 'EBITPS', 'CapitalSurplusFundPS', 'SurplusReserveFundPS',
                       'UndividedProfitPS', 'OperCashFlowPS', 'CashFlowPS', 'ShareholderFCFPS', 'EnterpriseFCFPS',
                       'DividendPS']
        back = 1
    return factor_list, select_list, back


'''
# 模式识别类===========================================================================================================
factor_list = ['CDLCONCEALBABYSWALL', 'CDLTHRUSTING', 'CDLMORNINGSTAR', 'CDLADVANCEBLOCK', 'CDLHIGHWAVE', 'CDLMATHOLD',
               'CDLSHOOTINGSTAR', 'CDLEVENINGDOJISTAR', 'CDLTAKURI', 'CDLSTALLEDPATTERN', 'CDLENGULFING',
               'CDLHIKKAKEMOD',
               'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLHAMMER', 'CDLPIERCING', 'CDLINVERTEDHAMMER', 'CDLSHORTLINE',
               'CDLKICKING',
               'CDLCOUNTERATTACK', 'CDLSPINNINGTOP', 'CDLSEPARATINGLINES', '/缺影线', 'CDLRICKSHAWMAN', 'CDLHOMINGPIGEON',
               'CDLKICKINGBYLENGTH', 'CDLINNECK', 'CDLONNECK', 'CDL2CROWS', 'CDLHARAMI', '/倒', 'CDLEVENINGSTAR',
               'CDL3STARSINSOUTH',
               'CDLUNIQUE3RIVER', 'CDLABANDONEDBABY', '/T', 'CDLIDENTICAL3CROWS', 'CDL3WHITESOLDIERS', 'CDL3INSIDE',
               'CDL3OUTSIDE',
               'CDL3LINESTRIKE', 'CDLTRISTAR', 'CDL3BLACKCROWS', 'CDLHANGINGMAN', '/下降三法', '/下降跳空三法', 'CDLDOJI',
               'CDLMORNINGDOJISTAR',
               'CDLDOJISTAR', 'CDLHARAMICROSS', 'CDLCLOSINGMARUBOZU', 'CDLLADDERBOTTOM', 'CDLSTICKSANDWICH',
               'CDLTASUKIGAP',
               'CDLBREAKAWAY', 'CDLDARKCLOUDCOVER', 'CDLHIKKAKE', 'CDLMATCHINGLOW', 'CDLUPSIDEGAP2CROWS', 'CDLBELTHOLD']

# 行业、分析师类========================================================================================================
factor_list = ['RSTR12', 'RSTR24', 'REC', 'DAREC', 'GREC', 'FY12P', 'DAREV', 'GREV', 'SFY12P', 'DASREV', 'GSREV',
               'EARNMOM', 'EPIBS',
               'SUE', 'SUOI', 'FEARNG', 'FSALESG', 'EgibsLong']

# 特色技术指标==========================================================================================================
factor_list = ['ADOSC', 'APO', 'AROONOSC', 'AVGPRICE', 'BOP', 'CORREL', 'DEMA', 'DX', 'KAMA', 'LINEARREG',
               'LINEARREG_ANGLE',
               'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'MAX', 'MAXINDEX', 'MEDPRICE', 'MIDPOINT', 'MIDPRICE', 'MIN',
               'MININDEX',
               'NATR', 'PPO', 'ROCP', 'ROCR', 'ROCR100', 'SAR', 'SMA', 'STDDEV', 'SUM', 'T3', 'TRIMA', 'TSF',
               'TYPPRICE', 'VAR',
               'WCLPRICE', 'WILLR', 'WMA', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_TRENDLINE', 'HT_TRENDMODE']
'''
