import pandas as pd
import numpy as np
import datetime as dt
from dateutil import parser

class Preprocessor():
    def __init__(self):
        self.xls = pd.ExcelFile('sp500 reg data 18-19.xlsx')
        # first sheet is blank
        self.stocks = self.xls.sheet_names[1:]
        self.etf_np = self.processETFs('etf 18-19.xlsx')
        self.input_data = self.Process()
        self.validification_data = self.Val()

    def Process(self):
        #writer = pd.ExcelWriter('processed.xlsx', engine='xlsxwriter')
        # output nparray with dates
        df = self.xls.parse(self.stocks[0], skiprows = 0)
        date_np = np.transpose(np.array(df))
        #out_np = [date_np[0][1:]]
        out_np = []

        for stock in self.stocks:
            # dataframe of data for 1 stock
            stk_df = self.xls.parse(stock, skiprows = 0)
            # remove dates
            stk_np = np.transpose(np.array(stk_df))[1:]
            stk_df = pd.DataFrame(np.transpose(stk_np))
            # interpolate and fill in missing vals
            for column in stk_df:
                stk_df[column] = pd.to_numeric(stk_df[column])
                stk_df[column] = stk_df[column].interpolate(method='linear', limit_direction= 'both', limit=3000, axis = 0)

            # np array of data
            stk_np = np.transpose(np.array(stk_df))

            #[0] Last Price
            out_np.append(np.transpose(self.pxLast(stk_np[0])))
            #[1] Current Market Cap
            out_np.append(np.transpose(self.marketCap(stk_np[1])))
            #[2] Price to Book Ratio
            out_np.append(np.transpose(self.mktToBook(stk_np[2])))
            #[3] RSI 14 day
            out_np.append(np.transpose(self.RSI(stk_np[3])))
            #[4] Best Target Price
            out_np.append(np.transpose(self.analyst(stk_np[4])))
            #[5] BEst Analyst Rating
            out_np.append(np.transpose(self.analyst(stk_np[5])))
            #[6] Volatility 30 Day
            out_np.append(np.transpose(self.vol(stk_np[6])))
            #[7] Volatility 90 Day
            out_np.append(np.transpose(self.vol(stk_np[7])))
            #[8] Normalized Accruals Balance Sheet Method
            out_np.append(np.transpose(stk_np[8][:-1]))
            #[9] Normalized Accruals Cash Flow Method
            out_np.append(np.transpose(stk_np[9][:-1]))
            #[10] Return on Assets
            out_np.append(np.transpose(stk_np[10][:-1]))
            #[11] Return on Common Equity
            out_np.append(np.transpose(stk_np[11][:-1]))
            #[12] Total Assets - 5 Yr Geometric Growth
            out_np.append(np.transpose(stk_np[12][:-1]))
            #[13] Price Earnings Ratio (P/E)
            out_np.append(np.transpose(self.peRatio(stk_np[13])))
            #[14] BEst P/E Next Year
            out_np.append(np.transpose(self.peRatio(stk_np[14])))
            #[15] Dividend 12 Month Yield
            out_np.append(np.transpose(stk_np[15][:-1]))
            #[16] Periodic EV to Trailing 12M EBITDA
            out_np.append(np.transpose(self.analyst(stk_np[16])))
            #[17] 3 Month Call Implied Volatility
            out_np.append(np.transpose(self.pxLast(stk_np[17])))
            #[18] 3 Month Put Implied Volatility
            out_np.append(np.transpose(self.vol(stk_np[18])))
            #[19] 30 Day Call Implied Volatility
            out_np.append(np.transpose(self.pxLast(stk_np[19])))
            #[20] 30 Day Put Implied Volatility
            out_np.append(np.transpose(self.vol(stk_np[20])))

        # append etf data
        for arr in self.etf_np:
          out_np.append(np.transpose(arr))

        #out_df = pd.DataFrame(np.transpose(out_np))
        #out_df.to_excel(writer)
        #writer.save()
        return np.transpose(out_np)

    # pt/pt-1 and then lagged (t=1 behind)
    def pxLast(self, price_arr):
        last = np.insert(price_arr, 0, 0)
        current = np.append(price_arr, 0)
        out = np.divide(current[1:-1], last[1:-1])
        return out

    # ln(market cap)
    def marketCap(self, mcap_arr):
        out = mcap_arr.astype(float)
        out = np.log(out)
        return out[:-1]

    # (market_to_bv)^-1
    def mktToBook(self, market_to_bv_arr):
        x = np.full_like(market_to_bv_arr, -1)
        return np.power(market_to_bv_arr,x)[:-1]

    # rsi*-1
    def RSI(self, rsi_arr):
        x = np.full_like(rsi_arr, -1)
        out = np.multiply(rsi_arr,x)
        return out[:-1]

    # ln(analysts)
    def analyst(self, analyst_arr):
        out = analyst_arr.astype(float)
        out = np.log(out)
        return out[:-1]

    # ln(vol t/vol t-1)- not lagged so a zero is inserted for t=0
    def vol(self, vol_arr):
        last = np.insert(vol_arr, 0, 0)
        current = np.append(vol_arr, 0)
        out = np.divide(current[1:-1], last[1:-1])
        out = out.astype(float)
        out = np.log(out)
        out = np.insert(out, 0,0)
        return out[:-1]

    # ln(peR t/peR t-1) - same as vol
    def peRatio(self, pe_arr):
        last = np.insert(pe_arr, 0, 0)
        current = np.append(pe_arr, 0)
        out = np.divide(current[1:-1], last[1:-1])
        out = out.astype(float)
        out = np.log(out)
        out = np.insert(out, 0,0)
        return out[:-1]

    def processETFs(self, xls):
        etf_xls = pd.ExcelFile(xls)
        df = etf_xls.parse(etf_xls.sheet_names[0], skiprows = 0)
        # remove dates
        s = np.transpose(np.array(df))[1:]
        df = pd.DataFrame(np.transpose(s))
        # interpolate and fill in missing vals
        for column in df:
            df[column] = pd.to_numeric(df[column])
            df[column] = df[column].interpolate(method='linear', limit_direction= 'both', limit=3000, axis = 0)
        etf_np = np.transpose(np.array(df))
        out_np = []

        for arr in etf_np:
            last = np.insert(arr, 0, 0)
            current = np.append(arr, 0)
            out = np.divide(current[1:-1], last[1:-1])
            out = out.astype(float)
            out = np.log(out)
            out_np.append(out)

        return out_np

    # this reduces input by 1 day because we validate current t with t+1 price info
    # one day is cut off the end of input data as well
    def Val(self):
        xls = pd.ExcelFile('sp500 08-19.xlsx')
        stk_df = xls.parse(xls.sheet_names[0], skiprows = 2611)
        # remove dates
        s = np.transpose(np.array(stk_df))[1:]
        df = pd.DataFrame(np.transpose(s))
        # interpolate and fill in missing vals
        for column in df:
            df[column] = pd.to_numeric(df[column])
            df[column] = df[column].interpolate(method='linear', limit_direction= 'both', limit=3000, axis = 0)
        # change to ln(t+1/t)
        data = np.transpose(np.array(df))
        val = []
        for arr in data:
            last = np.insert(arr, 0, 0)
            current = np.append(arr, 0)
            out = np.log(np.divide(current[1:], last[1:]))
            val.append(out)
        # transpose for daily row data in NN
        #return np.transpose(np.array(val))

        # [:-8 because different lengthed datasets right now]
        return np.transpose(np.array(val))[:-8]

    # ratio = train:test ratio
    def trainTestSplit(self, regData, valData, ratio):
        train_data = regData[:regData.shape[1]*ratio]
        test_data = regData[regData.shape[1]*ratio:]
        train_val = valData[:valData.shape[1] *ratio]
        test_val = valData[valData.shape[1] *ratio:]

        return train_data, test_data, train_val, test_val
