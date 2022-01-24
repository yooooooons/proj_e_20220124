#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import pyupbit
import datetime
import pandas as pd
import numpy as np
import warnings


# In[2]:


warnings.filterwarnings(action='ignore')   # 경고 메시지 비활성화, 활성화시엔 action=default 으로 설정


# In[3]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[4]:


access_key = "eziU49y9cSYp6BFEu8Vu8yEwk0AAZIxn1o0ya7Bp"
secret_key = "mjkWq13cmg1XE38l9xK7x80XhcIsyChHrmyx3IVe"


# In[5]:


coin_insert = [0, 79, 1]   # 첫번째 코인 종류(보통 비트코인)부터 몇번째 코인까지 시뮬레이션 하겠는가? (첫번째, 마지막 번째, 1)

check_currency = 'KRW'

except_no = 1   # 우선순위 코인 선정시, close_to_close 최대값 제외 행 수

succesive_rise_criteria = [3, 7, 1]
#dec_acept = [3, (succesive_rise_criteria[1] - 3), 1]
dec_acept = 2

candle_type = '15min'   # 사용하는 candle 종류 (1min, 3min, 5min, 10min, 15min, 30min, 60min)

rise_decision_ratio = 1.00007  # 이전 candle 대비 'close'값의 비율이 얼마 이상이면 상승으로 판단하는가 (예> 1 <-- '최근 close 값 / 이전값 close 값'이 1 이상이면 상승으로 판단)

candle_count = 100   # 투자시 최근 몇개 candle을 활용할 것인가 
simul_days = 2   # 며칠동안 시뮬레이션 할것인가?
simul_input_cash = 1000000   # 시뮬레이션시 투자 가정 금액

invest_ratio = 0.015   # 보유 금액의 최대 몇 % 를 투자할것인가 (예> 0.1 <-- 보유금액 10% 투자)
coin_interest_profit_ratio = 1.01   # 시뮬레이션 결과, 최근 이익율이 얼마 이상인것 대상으로 투자 검토할지 설정
dramatic_rise = 0.01   # 최근 n개 candle 기준, 한 candle에 (dramatic_rise * 100) % 이상 상승한 이력이 있으며, 매수 진행 안함
cut_off_open_price = 200   # open 가격이 얼마 이하인 것은 투자 대상에서 제외 (가격이 낮은것은 한 가격단위만 비싸게 구매해도 비율면에서 크게 올라감)

buy_margin = 0   # 몇 '거래단위' 가격 상승까지 매수가격으로 수용하여 매수하겠는가?   (예> 1 <-- BTC 거래단위인 5000원 x 1 단위까지 높은 가격으로 매수, 즉 현재가 75,000,000이면 75,005,000까지 매수)

sell_auto = 0.05   # 매수가 대비 몇 % 이상 하락했을때, candle 추이와 관계없이 자동 매도 하겠는가? (예> 0.02 <--- 2% 이상 상승시 자동 매도
sell_force = 0.03   # 매수가 대비 몇 % 이상 하락했을때, candle 추이와 관계없이 강제 매도 하겠는가? (예> 0.03 <--- 3% 이상 하락시 강제 매도
sell_one_candle_force = 0.01   # 한 candle 안에서 하락폭이 해당 candle의 open 가격 대비 일정 비율 보다 클때 강제 매도

transaction_fee_ratio = 0.0005   # 거래 수수료 비율

time_factor = 9   # 클라우드 서버 시차보정 (구글 클라우드 : time_factor = 9)


# ------------------------------------------------------------------------------------------------------------------------


decrease_penalty = 1   # 하락 candle에서 successive_rise 값을 몇이나 감소시킬 것인가
check_currency = 'KRW'
#check_coin_currency = invest_coin[4:]   # 'KRW-' 이후의 문자열, 코인 종류에 따라 3자리수, 4자리수 등 다양함

transaction_fee_ratio = 0.0005   # 매수 수수료 설정 (0.0005 <-- 수수료 0.05%)

if candle_type == '1min' :
    candle_adapt = 'minute1'
    time_unit = 1
elif candle_type == '3min' :
    candle_adapt = 'minute3'
    time_unit = 3
elif candle_type == '5min' :
    candle_adapt = 'minute5'
    time_unit = 5
elif candle_type == '10min' :
    candle_adapt = 'minute10'
    time_unit = 10
elif candle_type == '15min' :
    candle_adapt = 'minute15'
    time_unit = 15
elif candle_type == '30min' :
    candle_adapt = 'minute30'
    time_unit = 30
elif candle_type == '60min' :
    candle_adapt = 'minute60'
    time_unit = 60
elif candle_type == '240min' :
    candle_adapt = 'minute240'
    time_unit = 240
    
qualified_coin_day_count = int((1/3) * 24 * (60/time_unit)) + 1   #가장 최근 시간대를 제외해야 하기 때문에  + 1을 함

simul_candle_count = int(simul_days * (24 * (60/time_unit)))  # 최적 succesive_rise_No, dec_acept_No를 찾기위한 시뮬레이션에 사용하는 candle count 수
    
print ('candle_type : {0}  /  candle_adapt : {1}  /   candle_count : {2}'.format(candle_type, candle_adapt, candle_count))


# In[6]:


upbit = pyupbit.Upbit(access_key, secret_key)

balances = upbit.get_balances()
balances


# In[7]:


tickers = pyupbit.get_tickers()

LIST_coin_KRW = []

for i in range (0, len(tickers), 1):
    if tickers[i][0:3] == 'KRW':
        LIST_coin_KRW.append(tickers[i])
        
LIST_check_coin_currency = []

for i in range (0, len(LIST_coin_KRW), 1):
    LIST_check_coin_currency.append(LIST_coin_KRW[i][4:])


# In[8]:


# 잔고 조회, 현재가 조회 함수 정의

def get_balance(target_currency):   # 현급 잔고 조회
    """잔고 조회"""
    balances = upbit.get_balances()   # 통화단위, 잔고 등이 Dictionary 형태로 balance에 저장
    for b in balances:
        if b['currency'] == target_currency:   # 화폐단위('KRW', 'KRW-BTC' 등)에 해당하는 잔고 출력
            if b['balance'] is not None:
                return float(b['balance'])
            else:
                return 0
    return 0

def get_balance_locked(target_currency):   # 거래가 예약되어 있는 잔고 조회
    """잔고 조회"""
    balances = upbit.get_balances()   # 통화단위, 잔고 등이 Dictionary 형태로 balance에 저장
    for b in balances:
        if b['currency'] == target_currency:   # 화폐단위('KRW', 'KRW-BTC' 등)에 해당하는 잔고 출력
            if b['locked'] is not None:
                return float(b['locked'])
            else:
                return 0
    return 0

def get_avg_buy_price(target_currency):   # 거래가 예약되어 있는 잔고 조회
    """평균 매수가 조회"""
    balances = upbit.get_balances()   # 통화단위, 잔고 등이 Dictionary 형태로 balance에 저장
    for b in balances:
        if b['currency'] == target_currency:   # 화폐단위('KRW', 'KRW-BTC' 등)에 해당하는 잔고 출력
            if b['avg_buy_price'] is not None:
                return float(b['avg_buy_price'])
            else:
                return 0
    return 0


def get_current_price(invest_coin):
    """현재가 조회"""
    #return pyupbit.get_orderbook(tickers=invest_coin)[0]["orderbook_units"][0]["ask_price"]
    return pyupbit.get_current_price(invest_coin)

#price = pyupbit.get_current_price("KRW-BTC")


# In[9]:


# 투자 후보 종목 추출 함수 정의

factor_ratio_1 = 0.45
factor_ratio_2 = 0.1
factor_ratio_3 = 1 - (factor_ratio_1 + factor_ratio_2)

open_price_ratio_configure_ratio = 0.95
volume_price_configure_ratio = 0.5
No_of_rise_configure_ratio = 0.95

def qualified_coin () :
    print ('qualified_coin is checking')
    
    DF_basic_info_accum = pd.DataFrame({'coin_No' : ['temp'],   # 코인 번호
                                        'coin' : ['temp'],   # 코인 종류
                                        'open_price' : ['temp'],   # 코인 open 가격
                                        'open_price_ratio' : ['temp'],   # 최근일 마지막 open 가격 / 첫날 첫 open 가격
                                        'volume_price' : ['temp'],   # 거래량 / 최근일 마지막 open 가격
                                        'No_of_rise' : ['temp']})   # rise 횟수
    
    
    # 전체 코인 리스트의 평균 가격 변화율 / 거래량 / rise_candle 갯수를 초과하는 종목 추출
    for i in range (coin_insert[0], coin_insert[1] + coin_insert[2], coin_insert[2]) :
        DF_basic_info_intermediate = pyupbit.get_ohlcv(LIST_coin_KRW[i], count = qualified_coin_day_count, interval = candle_adapt)
        DF_basic_info_intermediate = DF_basic_info_intermediate.reset_index()
        DF_basic_info_intermediate_2 = DF_basic_info_intermediate.iloc[:-1]   # 이제 막 시작하여, 주기대비 일부시간만큼만 집계된 마지막 행은 제외
        #print(i)
        DF_basic_info_intermediate_2['coin_No'] = i
        #print ('DF_basic_info_intermediate_2[coin_No] :', DF_basic_info_intermediate_2['coin_No'])
        DF_basic_info_intermediate_2['coin'] = LIST_coin_KRW[i]
        #DF_basic_info_intermediate_2['open_price_ratio'] = DF_basic_info_intermediate_2['open'][len(DF_basic_info_intermediate_2) - 1] / DF_basic_info_intermediate_2['open'][0] 
        DF_basic_info_intermediate_2['open_price_ratio'] = DF_basic_info_intermediate_2['open'][int(round(len(DF_basic_info_intermediate_2) / 4))] / DF_basic_info_intermediate_2['open'][0]   # qualified_coin_day_countd의 가장 과거 결과와의 비교가 아닌, 비교적 최근 결과 기준으로 비율 산출
        #DF_basic_info_intermediate_2['std_ratio'] = DF_basic_info_intermediate_2['open'].std() / DF_basic_info_intermediate_2['open'][len(DF_basic_info_intermediate_2) - 1]
    
        DF_basic_info_intermediate_2['yester_close'] = DF_basic_info_intermediate_2['close'].shift(1)
        DF_basic_info_intermediate_2['yester_close'][0] = DF_basic_info_intermediate_2['close'][0]
        DF_basic_info_intermediate_2['close_to_close'] = DF_basic_info_intermediate_2['close'] / DF_basic_info_intermediate_2['yester_close']
        
        DF_close_to_close_descend = DF_basic_info_intermediate_2.sort_values(by = 'close_to_close', ascending = False)
        DF_basic_info_2 = DF_close_to_close_descend[except_no :]   # close_to_close 값이 가장 큰 n개 행은 제외
        
        DF_basic_info_2['volume_price'] = DF_basic_info_2['volume'] * DF_basic_info_2['open']
        DF_basic_info_2['volume_price'] = DF_basic_info_2['volume_price'].sum()
        
        DF_basic_info_2['open_price'] = DF_basic_info_2['open']
        
        DF_basic_info_2['rise'] = 0
        DF_basic_info_2.loc[DF_basic_info_2['close_to_close'] > rise_decision_ratio, 'rise'] = 1
        DF_basic_info_2['No_of_rise'] = DF_basic_info_2['rise'].sum()
        #print ('DF_basic_info_2\n' , DF_basic_info_2)
    
        DF_basic_info_use = DF_basic_info_2[['coin_No', 'coin', 'open_price', 'open_price_ratio', 'volume_price', 'No_of_rise']]
        #print ('DF_basic_info_use\n' , DF_basic_info_use.iloc[-2:])
    
        DF_basic_info_accum = pd.concat([DF_basic_info_accum, DF_basic_info_use.iloc[-2:-1]])   # 한행만 뽑으면 의도하지 않는 동작을 하여, 우선 두개행 추출
    
        time.sleep(1)
    
    DF_basic_info_selected = DF_basic_info_accum[DF_basic_info_accum['coin_No'] != 'temp']   # 'temp'가 입력된 행을 제외한 나머지 행들만 따로 추출
    DF_basic_info_selected = DF_basic_info_selected.drop_duplicates()   # 중복된 행 제거
    
    DF_basic_for_volume_check = DF_basic_info_selected.sort_values(by = 'volume_price', ascending = False).iloc[3:]   # BTC, ETH, XRP의 volume price가 항상 압도적이라서, 세개를 제외하고 평균 산출에 활용
    
    
    # 추출된 종목을 qualified = 1로 분류
    DF_basic_info_selected['qualified'] = 0
    #DF_basic_info_selected.iloc[0:1]['qulified'] = 1
    #DF_basic_info_selected['qulified'][1:2] = 1
    #DF_basic_info_selected['qulified'] = 1
    
    for i in range (0, len(DF_basic_info_selected) , 1) :
        #print (i)
        
        if ((DF_basic_info_selected.iloc[i]['open_price_ratio'] > (open_price_ratio_configure_ratio * DF_basic_info_selected['open_price_ratio'].mean())) &             (DF_basic_info_selected.iloc[i]['volume_price'] > (volume_price_configure_ratio * DF_basic_for_volume_check['volume_price'].mean())) &             (DF_basic_info_selected.iloc[i]['No_of_rise'] > (No_of_rise_configure_ratio * DF_basic_info_selected['No_of_rise'].mean())) &             (DF_basic_info_selected.iloc[i]['open_price'] > cut_off_open_price)) :
            DF_basic_info_selected.iloc[i:i+1]['qualified'] = 1
    
    #print ('DF_basic_info_selected[open_price_ratio] : {0}  / volume_price_configure_ratio * DF_basic_info_selected[volume_price] : {1}  / DF_basic_info_selected[No_of_rise] : {2}'.format(DF_basic_info_selected['open_price_ratio'].mean(), (volume_price_configure_ratio * DF_basic_for_volume_check['volume_price'].mean()), DF_basic_info_selected['No_of_rise'].mean()))
    print ('\nDF_basic_info_selected\n', DF_basic_info_selected)
    
    DF_candidate = DF_basic_info_selected.loc[DF_basic_info_selected['qualified'] == 1]
    print ('\nqualified_DF_candidate\n', DF_candidate)
    
    # qualified 종목 중, 우선순위 정하기
    DF_candidate['potential_factor'] = (factor_ratio_1 * (DF_candidate['open_price_ratio'] / DF_basic_info_selected['open_price_ratio'].mean()))     + (factor_ratio_2 * (DF_candidate['volume_price'] / DF_basic_for_volume_check['volume_price'].mean()))     + (factor_ratio_3 * (DF_candidate['No_of_rise'] / DF_basic_info_selected['No_of_rise'].mean()))
    
    DF_candidate_descend = DF_candidate.sort_values(by = 'potential_factor', ascending = False)
    #print ('\nDF_candidate_descend\n', DF_candidate_descend)
    
    
    return DF_candidate_descend
    
    


# In[10]:


# DF candle 오름/하락 tagging용 함수 정의

def rise_tagging (DF_input, successive_rise_No) :
    #print('rise_tagging is conducting')
    
    DF_input['yester_close'] = DF_input['close'].shift(1)
    DF_input['yester_close'][0] =  DF_input['close'][0]
    DF_input['successive_rise'] = 0
             
    DF_input['rise'] = 0
    DF_input['close_to_close'] = DF_input['close'] / DF_input['yester_close']
    
    DF_input.loc[DF_input['close_to_close'] > rise_decision_ratio, 'rise'] = 1
    
    buying_interrupt = 0
    
    #print('\nDF_input', DF_input)
    
    for i in range (1, len(DF_input), 1) :
        if DF_input['rise'][i] == 0 :
            if DF_input['successive_rise'][i-1] <= decrease_penalty :
                DF_input['successive_rise'][i] = 0
            elif DF_input['successive_rise'][i-1] > decrease_penalty :
                DF_input['successive_rise'][i] = DF_input['successive_rise'][i-1] - decrease_penalty
                        
        elif DF_input['rise'][i] == 1 :
            if DF_input['successive_rise'][i-1] < successive_rise_No :
                DF_input['successive_rise'][i] = DF_input['successive_rise'][i-1] + 1
            elif DF_input['successive_rise'][i-1] >= successive_rise_No :
                DF_input['successive_rise'][i] = DF_input['successive_rise'][i-1]
                
    
    # 매수 대상 제외 조건 부합여부 점검
    if (DF_input['successive_rise'][-3] == successive_rise_No) | (DF_input['successive_rise'][-4] == successive_rise_No) | (DF_input['successive_rise'][-5] == successive_rise_No) :   # 최근 이전에 이미 buying 조건을 달성한 이력이 있으면 매수 진행 안함(첫 매수에서 발생)
        buying_interrupt = 1
    
    #if ((DF_input['close_to_close'][-2] > (1 + dramatic_rise)) | (DF_input['close_to_close'][-3] > (1 + dramatic_rise)) | (DF_input['close_to_close'][-4] > (1 + dramatic_rise))) :   # 최근 급등한 candle이 있으면 매수 진행 안함
    if (DF_input['close_to_close'][-2] > (1 + dramatic_rise)) :   # 최근 급등한 candle이 있으면 매수 진행 안함
        buying_interrupt = buying_interrupt + 2
    
    #print ('\nDF_input_last 10 rows\n', DF_input.iloc[-10:])
    
    DF_input.to_excel('DF_input.xlsx')
    
    #return DF_input['successive_rise'][-2]   #가장 마지막행값[-1]은 계속 변하는 값. 따라서 완료된 값중 가장 최신인[-2] 전달
    return DF_input, buying_interrupt
    


# In[11]:


# 최적의 succesive_rise_criteria, dec_acept 조건 도출 함수 정의

def proper_rise_dec_condition (candidate_coin) :
    print ('proper_rise_dec_condition for [ {0} ] is finding'.format(candidate_coin))
    
    #first_profit_ratio = 0.003   # 3연속 하락을 통해 매수했을때만 고려시 매도 이익율
    #other_profit_ratio = 0.002   # 그외 하락을 통해 매수했을때, 매도 이익율
    
    buy_price_upside_ratio = 0   # 매수시 매수시점 가격대비, 실제 매수가 가정 상승율
    sell_price_downside_ratio = 0   # 매도시 매도시점 가격대비, 실제 매도가 가정 하락율
    
    DF_sim_accum = pd.DataFrame({'coin' : ['temp'],
                                 'succesive_rise_No' : ['temp'],
                                 'dec_acept_No' : ['temp'],
                                 'total_cash' : ['temp'],
                                 'total_value' : ['temp'],
                                 'Not_buy_condition' : ['temp']}) 
    
    
    DF_import = pyupbit.get_ohlcv(candidate_coin, count = simul_candle_count, interval = candle_adapt)
    
    for i in range (succesive_rise_criteria[0], succesive_rise_criteria[1] + succesive_rise_criteria[2], succesive_rise_criteria[2]) :
        #for k in range (dec_acept[0], dec_acept[1] + dec_acept[2], dec_acept[2]) :
        for k in range (i-1, dec_acept - 1, -1) :
            #print ('succesive_rise_No : {0}  / dec_acept_No : {1}'.format(i, k))
            
            total_cash = simul_input_cash   # 시뮬레이션용 가정 예산
            initial_total_cash = total_cash
            sim_invest_ratio = 1   # 보유 금액의 몇 % 를 투자할것인가 (예> 0.1 <-- 보유금액 10% 투자)
            
            bought_price = 0
            bought_volume = 0
            sell_price = 0
            sell_volume = 0
            bought_state = 0
            bought_timing_rise_value = 0
            
            DF_candidate = DF_import.copy()
            DF_input_candidate, not_buy_condition_check = rise_tagging (DF_candidate, i)
            
            DF_input_candidate['coin'] = candidate_coin
            DF_input_candidate['succesive_rise_No'] = i
            DF_input_candidate['dec_acept_No'] = k
            DF_input_candidate['total_value'] = 0
            DF_input_candidate['total_cash'] = 0
            DF_input_candidate['transaction'] = 'none'
            DF_input_candidate['bought_state'] = 0
            DF_input_candidate['Not_buy_condition'] = not_buy_condition_check
            #print ('\nDF_input_candidate', DF_input_candidate)
            
            for m in range (i, len(DF_input_candidate), 1) :
                DF_input_candidate['total_cash'][0:i] = total_cash
                DF_input_candidate['total_value'][0:i] = total_cash
                
                if bought_state == 0 :
                    if (DF_input_candidate.iloc[m-2]['successive_rise'] == i - 1) & (DF_input_candidate.iloc[m-1]['successive_rise'] == i) :
                        #print ('buying_trial__m : {0}  / i - 1 : {1} / i : {2}'.format(m, (i - 1), i))
                        bought_price = DF_input_candidate['open'][m] * (1 + buy_price_upside_ratio)
                        bought_volume = ((total_cash * sim_invest_ratio) * (1 - transaction_fee_ratio)) / bought_price
                        total_cash = total_cash - (bought_price * bought_volume) - (bought_price * bought_volume * transaction_fee_ratio)
                        DF_input_candidate['transaction'][m] = 'buy'
                        DF_input_candidate['bought_state'][m] = bought_state
                        DF_input_candidate['total_cash'][m] = total_cash
                        DF_input_candidate['total_value'][m] = bought_price * bought_volume
                        bought_state = 1
                    else : 
                        DF_input_candidate['bought_state'][m] = bought_state
                        DF_input_candidate['total_cash'][m] = DF_input_candidate['total_cash'][m-1]
                        DF_input_candidate['total_value'][m] = DF_input_candidate['total_value'][m-1]                 
                
                else :
                    #if (DF_inform_use.iloc[m-2]['successive_rise'] >= succesive_rise_criteria) & ((DF_inform_use.iloc[m-1]['successive_rise'] <= succesive_rise_criteria - 1) | (DF_inform_use.iloc[m-1]['successive_rise'] <= succesive_rise_criteria - 2)) :
                    if (DF_input_candidate.iloc[m-1]['successive_rise'] == i - k) :
                        #print ('selling_trial__m :{0}  / i - k : {1}'.format(m , (i - k)))
                        sell_price = DF_input_candidate['open'][m] * (1 - sell_price_downside_ratio)
                        sell_volume = bought_volume
                        total_cash = sell_price * bought_volume * (1 - transaction_fee_ratio)
                        DF_input_candidate['transaction'][m] = 'sell'
                        DF_input_candidate['bought_state'][m] = bought_state
                        DF_input_candidate['total_cash'][m] = total_cash
                        DF_input_candidate['total_value'][m] = total_cash
                        bought_state = 0
                    else :
                        DF_input_candidate['bought_state'][m] = bought_state
                        DF_input_candidate['total_cash'][m] = DF_input_candidate['total_cash'][m-1]
                        DF_input_candidate['total_value'][m] = DF_input_candidate['open'][m] * bought_volume
                                   
            #DF_input_candidate.to_excel('DF_input_candidate_{0}_{1}.xlsx'.format(i, k))
            DF_candidate_selected = DF_input_candidate[['coin','succesive_rise_No', 'dec_acept_No', 'total_cash', 'total_value', 'Not_buy_condition']]
            
            DF_sim_accum = pd.concat([DF_sim_accum, DF_candidate_selected.iloc[-1:]])
    
    DF_sim_accum_final = DF_sim_accum[DF_sim_accum['coin'] != 'temp']   # 'temp'가 입력된 행을 제외한 나머지 행들만 따로 추출
    DF_sim_accum_descend = DF_sim_accum_final.sort_values(by = 'total_value', ascending = False)
    #DF_sim_accum_descend = DF_sim_accum_final.sort_values(by = 'total_cash', ascending = False)
    #print ('\DF_sim_accum_descend', DF_sim_accum_descend)
    DF_sim_accum_descend.to_excel('DF_sim_accum_descend.xlsx')
    
    
    return DF_sim_accum_descend


# In[12]:


bought_state = 0


# In[13]:


# 매수 대상 코인 및 successive_rise_No / dec_acept_No 후보 산출하기

def target_coin_with_rise_No () :
    print ('target_coin with successive_rise_No and dec_acept_No is finding')
    
    DF_selected_coin = qualified_coin ()
    LIST_candidate_coin = []
    for i in range (0, len(DF_selected_coin), 1) :
        LIST_candidate_coin.append(DF_selected_coin.iloc[i]['coin'])
    print ('LIST_candidate_coin : ', LIST_candidate_coin)
    
    accum_finding_target_coin = pd.DataFrame({'coin' : ['temp'],
                                              'succesive_rise_No' : ['temp'],
                                              'dec_acept_No' : ['temp'],
                                              'total_cash' : ['temp'],
                                              'total_value' : ['temp'],
                                              'Not_buy_condition' : ['temp']})
      
    for k in LIST_candidate_coin :
        DF_intermidiate_check = proper_rise_dec_condition (k)
        DF_intermidiate2_check = DF_intermidiate_check[['coin','succesive_rise_No', 'dec_acept_No', 'total_cash', 'total_value', 'Not_buy_condition']]
        accum_finding_target_coin = pd.concat([accum_finding_target_coin, DF_intermidiate2_check])
    
    print ('\naccum_finding_target_coin\n', accum_finding_target_coin)
    
    accum_finding_target_coin_filtered = accum_finding_target_coin[accum_finding_target_coin['coin'] != 'temp']   # 'temp'가 입력된 행을 제외한 나머지 행들만 따로 추출
    accum_finding_target_coin_filtered = accum_finding_target_coin_filtered[accum_finding_target_coin_filtered['Not_buy_condition'] <= 1]   # not_buying 조건이 1이 아닌 경우만 검토 대상으로 선별
    
    accum_finding_target_coin_descend_value = accum_finding_target_coin_filtered.sort_values(by = 'total_value', ascending = False)
    print ('\naccum_finding_target_coin_descend_value\n', accum_finding_target_coin_descend_value)
    #accum_finding_target_coin_descend_cash = accum_finding_target_coin_filtered.sort_values(by = 'total_cash', ascending = False)
    #print ('\naccum_finding_target_coin_descend_by_cash\n', accum_finding_target_coin_descend_cash)
    #print ('\naccum_finding_target_coin_descend_by_cash\n')
    #accum_finding_target_coin_descend_cash

    accum_finding_target_coin_descend_value_interested = accum_finding_target_coin_descend_value[accum_finding_target_coin_descend_value['total_value'] >= (coin_interest_profit_ratio * simul_input_cash)]
    print ('\naccum_finding_target_coin_descend_value_interested\n', accum_finding_target_coin_descend_value_interested)

    #accum_finding_target_coin_descend_value = accum_finding_target_coin_filtered.sort_values(by = 'total_value', ascending = False)
    #print ('\naccum_finding_target_coin_descend_by_value\n', accum_finding_target_coin_descend_value)
    #print ('\naccum_finding_target_coin_descend_by_value\n')
    #accum_finding_target_coin_descend_value

    #accum_finding_target_coin_descend.to_excel('accum_finding_target_coin_descend.xlsx')
    
    #check_coin_currency = target_coin[4:]   # 'KRW-' 이후의 문자열, 코인 종류에 따라 3자리수, 4자리수 등 다양함
    
    LIST_invest_target_coin = []
    for m in range (0, len(accum_finding_target_coin_descend_value_interested), 1) :
        print ('m :', m)
        intmediate_LIST_target_coin = []
        intmediate_LIST_target_coin.append(accum_finding_target_coin_descend_value_interested.iloc[m]['coin'])
        intmediate_LIST_target_coin.append(accum_finding_target_coin_descend_value_interested.iloc[m]['succesive_rise_No'])
        intmediate_LIST_target_coin.append(accum_finding_target_coin_descend_value_interested.iloc[m]['dec_acept_No'])
        
        LIST_invest_target_coin.append(intmediate_LIST_target_coin)
    
    print ('\nLIST_invest_target_coin : ', LIST_invest_target_coin)
                                            
    
    return accum_finding_target_coin_descend_value_interested, LIST_invest_target_coin
    


# In[14]:


temp_DF_preliminary_target_coin, temp_LIST_preliminary_target_coin = target_coin_with_rise_No ()


# In[15]:


global bought_coin
global bought_opt_successive_rise_No
global bought_dec_acept_No
global check_coin_currency
#global target_coin
#global opt_succesive_rise_No

bought_coin = temp_LIST_preliminary_target_coin[0][0]
bought_opt_successive_rise_No = temp_LIST_preliminary_target_coin[0][1]
bought_dec_acept_No = temp_LIST_preliminary_target_coin[0][2]
check_coin_currency = temp_LIST_preliminary_target_coin[0][0][4:]
#target_coin = temp_LIST_preliminary_target_coin[0][0]
#opt_succesive_rise_No = temp_LIST_preliminary_target_coin[0][1]


# In[16]:


#LIST_preliminary_target_coin[1]
#LIST_preliminary_target_coin[1][0]
#len(LIST_preliminary_target_coin)


# In[ ]:





# In[17]:



while True:
    
    try:
        
        now = datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600))   # 클라우드 서버와 한국과의 시간차이 보정 (9시간)
        print ('bought_state : {0}   / now : {1}'.format(bought_state, now))
        
        
        if (now.minute % time_unit == 0) & (52 < (now.second % 60) <= 57) :   # N시:00:02초 ~ N시:00:07초 사이 시각이면
            balances2 = upbit.get_balances()
            print ('current_aseet_status\n', balances2)
 
         
        # 매수 영역
        if (bought_state == 0) & ((now.minute % time_unit == 0) & (52 < (now.second % 60) <= 57)) :   # N시:00:02초 ~ N시:00:07초 사이 시각이면
            
            DF_preliminary_target_coin, LIST_preliminary_target_coin = target_coin_with_rise_No ()
            #print ('LIST_preliminary_target_coin\n :', LIST_preliminary_target_coin)
            
            investing_trial_No = 0
            while (investing_trial_No < len(LIST_preliminary_target_coin)) :
                
                DF_last = pyupbit.get_ohlcv(LIST_preliminary_target_coin[investing_trial_No][0], count = candle_count, interval = candle_adapt)
                DF_check, buying_inter = rise_tagging(DF_last, LIST_preliminary_target_coin[investing_trial_No][1])
                #print ('DF_check\n' , DF_check)
                last_succesive_rise_No = DF_check['successive_rise'][-2]
                print ('investing_trial_No : {0}_____coin : {1}  / successive_rise_No : {2}  / dec_acept_No : {3}  ==> last_succesive_rise_No : {4} '.format(investing_trial_No, LIST_preliminary_target_coin[investing_trial_No][0], LIST_preliminary_target_coin[investing_trial_No][1], LIST_preliminary_target_coin[investing_trial_No][2], last_succesive_rise_No))
             
                if (last_succesive_rise_No == LIST_preliminary_target_coin[investing_trial_No][1]) :
                    
                    print ('$$$$$ buying_transaction is coducting $$$$$')
                    bought_coin = LIST_preliminary_target_coin[investing_trial_No][0]
                    bought_opt_successive_rise_No = LIST_preliminary_target_coin[investing_trial_No][1]
                    bought_dec_acept_No = LIST_preliminary_target_coin[investing_trial_No][2]
                    check_coin_currency = LIST_preliminary_target_coin[investing_trial_No][0][4:]
                    #target_coin = LIST_preliminary_target_coin[investing_trial_No][0]
                    #opt_succesive_rise_No = LIST_preliminary_target_coin[investing_trial_No][1]
                
                    investing_trial_No = investing_trial_No + len(LIST_preliminary_target_coin)   # 매수 조건을 만족하므로, 더이상 다른 코인 / rise_NO / dec_No 만족여부 확인이 불필요 하므로, 이번 턴만 하고 빠져나감
                    
                    temp_current_price = get_current_price(bought_coin)
                    
                    if temp_current_price >= 1000000 :  # 200만원 이상은 거래단위가 1000원, 100~200만원은 거래단위가 500원이지만 편의상 200만원 이상과 함께 처리
                        unit_factor = -3
                        unit_value = 1000
                    elif temp_current_price >= 100000 :
                        unit_factor = -2
                        unit_value = 50
                    elif temp_current_price >= 10000 :
                        unit_factor = -1
                        unit_value = 10
                    elif temp_current_price >= 1000 :
                        unit_factor = -1
                        unit_value = 5
                    elif temp_current_price >= 100 :
                        unit_factor = 0
                        unit_value = 1
                    else :
                        temp_current_price <= 100   # 100원 미만은 별도로 code에서 int형이 아닌 float형으로 형변환 해줘야함
                        unit_factor = 1
                        unit_value = 0.1
                    
                    
                    bought_price = temp_current_price + ((buy_margin -1) * unit_value)
                    investable_budget = get_balance(check_currency) * invest_ratio
                    bought_volume = (investable_budget * (1 - transaction_fee_ratio)) / bought_price
                    transaction_buy = upbit.buy_limit_order(bought_coin, bought_price, bought_volume)
                    time.sleep(5)
                    print ('buy_transaction_result :', transaction_buy)
                    print ('bought_target_volume : {0}  /  bought_volume_until_now : {1}  /  now_in buying_check mode : {1}'.format(bought_volume, get_balance(check_coin_currency), (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600)))))
                    buy_time = datetime.datetime.now().hour
                    time.sleep(10)
            
                    while ((datetime.datetime.now().minute % time_unit) < (time_unit -1)) :   # 한번에 매수 물량 전체가 매수가 안될것을 고려하여, 1 time unit 동안은 매수 시도 유지
                        print ('bought_target_volume : {0}  /  bought_volume_until_now : {1}  /  now_in buying_check mode : {1}'.format(bought_volume, get_balance(check_coin_currency), (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600)))))
                        #print ('bought_state : {0}  /  now_in buiyng_loop : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600)))))
                        
                        if get_balance(check_coin_currency) >= (0.9 * bought_volume) :
                            print ('bought_target_volume is (almost) bought')
                            break   # 매수 계획 물량이 실제 매수 되었으면 while 문 탈출
                        
                        if get_current_price(bought_coin) <= bought_price * (1-sell_one_candle_force) :   # 만약 매수 시도 시간(1 time_unit) 중간에, 하락 수용 가능 수준 이상으로 하락하게 되면
                            transaction_sell = upbit.sell_market_order(bought_coin, get_balance(check_coin_currency))   # 시장가에 매도
                            print ('\nnow :', (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600))))
                            print ('sell_transaction_result :', transaction_sell)
                            bought_state = 0
                            
                            break   # 강제 매도시 while 문 탈출
                            time.sleep(5)
                            
                
                    time.sleep(3)
            
                    print ('transaction_buy_cancel is considering')
                    transaction_buy_cancel = upbit.cancel_order(transaction_buy['uuid'])   # 1시간 매수 시간 동안에도 매수가 미수에 그치면 매수 중단
                    time.sleep(10)
                    print ('transaction_buy_cancel is considering')
                    transaction_buy_cancel = upbit.cancel_order(transaction_buy['uuid'])   # 1시간 매수 시간 동안에도 매수가 미수에 그치면 매수 중단
                
                else :
                    investing_trial_No = investing_trial_No + 1   # 다른 코인 / rise_NO / dec_No 만족여부 확인을 위해 숫자 1 증가
                    
    
        if get_balance(check_coin_currency) == 0 :
            bought_state = 0
            print ('bought_state is 0')
        
        if get_balance(check_coin_currency) > 0 :
            bought_state = 1
            print ('bought_state is 1')

 
    
        # 일반 매도 영역

        if (bought_state == 1) :
            if (now.minute % time_unit == 0) & (52 < (now.second % 60) <= 57) :
                #DF_last = pyupbit.get_ohlcv(bought_coin, count = candle_count, interval = candle_adapt)
                DF_last = pyupbit.get_ohlcv(bought_coin, count = candle_count, interval = candle_adapt)
                DF_check, buying_inter2 = rise_tagging(DF_last, bought_opt_successive_rise_No)
                last_succesive_rise_No = DF_check['successive_rise'][-2]
                print ('last_succesive_rise_No : ', last_succesive_rise_No)
            
                if last_succesive_rise_No == (bought_opt_successive_rise_No - bought_dec_acept_No) :   # 기준 수준 이상으로 하락이 연속되면
                    transaction_sell = upbit.sell_market_order(bought_coin, get_balance(check_coin_currency))   # 시장가에 매도
                    print ('bought_state : {0}  / now_in selling_check mode : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600)))))
                    print ('sell_transaction_result\n :', transaction_sell)
                    time.sleep(time_unit * 60)
                    bought_state = 0
                
                
        # 특이 매도 영역
        # 1) 매수가 대비 하락폭이 일정 비율 보다 클때 강제 매도     
        if (bought_state == 1) :
            if (get_current_price(bought_coin) <= bought_price * (1-sell_force)) :   # 하락폭이 기준수준보다 크다면
                transaction_sell = upbit.sell_market_order(bought_coin, get_balance(check_coin_currency))   # 시장가에 매도
                print ('bought_state : {0}  / now_in FORCED selling_check mode : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600)))))
                print ('sell_transaction_result :', transaction_sell)
                time.sleep(time_unit * 60)
                bought_state = 0
            
        # 2) 한 candle 안에서 하락폭이 해당 candle의 open 가격 대비 일정 비율 보다 클때 강제 매도
        if (bought_state == 1) :
            DF_one_candle_check = pyupbit.get_ohlcv(bought_coin, count = 5, interval = candle_adapt)
            if (get_current_price(bought_coin) <= ((1 - sell_one_candle_force) * DF_one_candle_check.iloc[-1]['open'])) :
                transaction_sell = upbit.sell_market_order(bought_coin, get_balance(check_coin_currency))   # 시장가에 매도
                print ('bought_state : {0}  / now_in one_candle FORCED selling_check mode : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600)))))
                print ('sell_transaction_result :', transaction_sell)
                time.sleep(time_unit * 60)
                bought_state = 0
            
        # 3) 매수가 대비 상승폭이 일정 비율 보다 클때 자동 매도       
        if (bought_state == 1) :
            if (get_current_price(bought_coin) >= bought_price * (1 + sell_auto)) :   # 상승폭이 기준수준보다 크다면
                transaction_sell = upbit.sell_market_order(bought_coin, get_balance(check_coin_currency))   # 시장가에 매도
                print ('bought_state : {0}  / now_in AUTO selling_check mode : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600)))))
                print ('sell_transaction_result :', transaction_sell)
                time.sleep(time_unit * 60)
                bought_state = 0
    
        time.sleep(1)
    
    
    
    except Exception as e:
        print('ERROR')
        time.sleep(1)


# In[ ]:


'''
while True:
    
    try :
        now = datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600))   # 클라우드 서버와 한국과의 시간차이 보정 (9시간)
        print ('bought_state : {0}   / now : {1}'.format(bought_state, now))
 
         
        # 매수 영역
        if (bought_state == 0) & ((now.minute % time_unit == 0) & (52 < (now.second % 60) <= 57)) :   # N시:00:02초 ~ N시:00:07초 사이 시각이면
            
            DF_preliminary_target_coin, LIST_preliminary_target_coin = target_coin_with_rise_No ()
            
            investing_trial_No = 0
            while (investing_trial_No < len(LIST_preliminary_target_coin)) :
                
                DF_last = pyupbit.get_ohlcv(LIST_preliminary_target_coin[investing_trial_No][0], count = candle_count, interval = candle_adapt)
                DF_check, buying_inter = rise_tagging(DF_last, LIST_preliminary_target_coin[investing_trial_No][1])
                #print ('DF_check\n' , DF_check)
                last_succesive_rise_No = DF_check['successive_rise'][-2]
                print ('invest_No : {0}_____coin : {1}  / successive_rise_No : {2}  / dec_acept_No : {3}  ==> last_succesive_rise_No : {4} '.format(investing_trial_No, LIST_preliminary_target_coin[investing_trial_No][0], LIST_preliminary_target_coin[investing_trial_No][1], LIST_preliminary_target_coin[investing_trial_No][2], last_succesive_rise_No))
             
                if (last_succesive_rise_No == LIST_preliminary_target_coin[investing_trial_No][1]) :
                    investing_No = investing_No + len(LIST_preliminary_target_coin)   # 매수 조건을 만족하므로, 더이상 다른 코인 / rise_NO / dec_No 만족여부 확인이 불필요 하므로, 이번 턴만 하고 빠져나감
                    
                    print ('$$$$$ buying_transaction is coducting $$$$$')
                    check_coin_currency = LIST_preliminary_target_coin[investing_No][0][4:]
                    target_coin = LIST_preliminary_target_coin[investing_No][0]
                    opt_succesive_rise_No = LIST_preliminary_target_coin[investing_No][1]
                    
                    temp_current_price = get_current_price(target_coin)
                    
                    if temp_current_price >= 1000000 :  # 200만원 이상은 거래단위가 1000원, 100~200만원은 거래단위가 500원이지만 편의상 200만원 이상과 함께 처리
                        unit_factor = -3
                        unit_value = 1000
                    elif temp_current_price >= 100000 :
                        unit_factor = -2
                        unit_value = 50
                    elif temp_current_price >= 10000 :
                        unit_factor = -1
                        unit_value = 10
                    elif temp_current_price >= 1000 :
                        unit_factor = -1
                        unit_value = 5
                    elif temp_current_price >= 100 :
                        unit_factor = 0
                        unit_value = 1
                    else :
                        temp_current_price <= 100   # 100원 미만은 별도로 code에서 int형이 아닌 float형으로 형변환 해줘야함
                        unit_factor = 1
                        unit_value = 0.1
                    
                    
                    bought_price = temp_current_price + ((buy_margin -1) * unit_value)
                    investable_budget = get_balance(check_currency) * invest_ratio
                    bought_volume = (investable_budget * (1 - transaction_fee_ratio)) / bought_price
                    transaction_buy = upbit.buy_limit_order(target_coin, bought_price, bought_volume)
                    time.sleep(5)
                    print ('buy_transaction_result :', transaction_buy)
                    print ('bought_target_volume : {0}  /  bought_volume_until_now : {1}  /  now_in buying_check mode : {1}'.format(bought_volume, get_balance(check_coin_currency), (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600)))))
                    buy_time = datetime.datetime.now().hour
                    time.sleep(10)
            
                    while ((datetime.datetime.now().minute % time_unit) < (time_unit -1)) :   # 한번에 매수 물량 전체가 매수가 안될것을 고려하여, 1 time unit 동안은 매수 시도 유지
                        print ('bought_target_volume : {0}  /  bought_volume_until_now : {1}  /  now_in buying_check mode : {1}'.format(bought_volume, get_balance(check_coin_currency), (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600)))))
                        #print ('bought_state : {0}  /  now_in buiyng_loop : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600)))))
                
                        if get_current_price(target_coin) <= bought_price * (1-sell_one_candle_force) :   # 만약 매수 시도 시간(1 time_unit) 중간에, 하락 수용 가능 수준 이상으로 하락하게 되면
                            transaction_sell = upbit.sell_market_order(target_coin, get_balance(check_coin_currency))   # 시장가에 매도
                            print ('\nnow :', (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600))))
                            print ('sell_transaction_result :', transaction_sell)
                
                    time.sleep(1)
            
                    print ('transaction_buy_cancel is considering')
                    transaction_buy_cancel = upbit.cancel_order(transaction_buy['uuid'])   # 1시간 매수 시간 동안에도 매수가 미수에 그치면 매수 중단
                    
                
                else :
                    investing_trial_No = investing_trial_No + 1   # 다른 코인 / rise_NO / dec_No 만족여부 확인을 위해 숫자 1 증가
                    
    
        if get_balance(check_coin_currency) == 0 :
            bought_state = 0
            print ('bought_state is 0')
        
        if get_balance(check_coin_currency) > 0 :
            bought_state = 1
            print ('bought_state is 1')
 
    
        # 일반 매도 영역

        if (bought_state == 1) :
            if (now.minute % time_unit == 0) & (52 < (now.second % 60) <= 57) :
                DF_last = pyupbit.get_ohlcv(target_coin, count = candle_count, interval = candle_adapt)
                DF_check = rise_tagging(DF_last, opt_succesive_rise_No)
                last_succesive_rise_No = DF_check['successive_rise'][-2]
                print ('last_succesive_rise_No : ', last_succesive_rise_No)
            
                if last_succesive_rise_No == (opt_succesive_rise_No - opt_dec_acept_No) :   # 기준 수준 이상으로 하락이 연속되면
                    transaction_sell = upbit.sell_market_order(target_coin, get_balance(check_coin_currency))   # 시장가에 매도
                    print ('bought_state : {0}  / now_in selling_check mode : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600)))))
                    print ('sell_transaction_result :', transaction_sell)
                    time.sleep(10)
                
        # 특이 매도 영역
        # 1) 매수가 대비 하락폭이 일정 비율 보다 클때 강제 매도     
        if (bought_state == 1) :
            if (get_current_price(target_coin) <= bought_price * (1-sell_force)) :   # 하락폭이 기준수준보다 크다면
                transaction_sell = upbit.sell_market_order(target_coin, get_balance(check_coin_currency))   # 시장가에 매도
                print ('bought_state : {0}  / now_in FORCED selling_check mode : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600)))))
                print ('sell_transaction_result :', transaction_sell)
                time.sleep(10)
            
        # 2) 한 candle 안에서 하락폭이 해당 candle의 open 가격 대비 일정 비율 보다 클때 강제 매도
        if (bought_state == 1) :
            DF_one_candle_check = pyupbit.get_ohlcv(target_coin, count = 5, interval = candle_adapt)
            if (get_current_price(target_coin) <= ((1 - sell_one_candle_force) * DF_one_candle_check.iloc[-1]['open'])) :
                transaction_sell = upbit.sell_market_order(target_coin, get_balance(check_coin_currency))   # 시장가에 매도
                print ('bought_state : {0}  / now_in one_candle FORCED selling_check mode : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600)))))
                print ('sell_transaction_result :', transaction_sell)
                time.sleep(10)
            
        # 3) 매수가 대비 상승폭이 일정 비율 보다 클때 자동 매도       
        if (bought_state == 1) :
            if (get_current_price(target_coin) >= bought_price * (1 + sell_auto)) :   # 상승폭이 기준수준보다 크다면
                transaction_sell = upbit.sell_market_order(target_coin, get_balance(check_coin_currency))   # 시장가에 매도
                print ('bought_state : {0}  / now_in AUTO selling_check mode : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600)))))
                print ('sell_transaction_result :', transaction_sell)
                time.sleep(10)
    
        time.sleep(1)
    
    
    except Exception as e:
        print('ERROR')
        time.sleep(1)
'''


# In[ ]:


'''
while True:
    
    try :
        now = datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600))   # 클라우드 서버와 한국과의 시간차이 보정 (9시간)
        print ('bought_state : {0}   / now : {1}'.format(bought_state, now))
        
        # 매수 영역
        if (bought_state == 0) & ((now.minute % time_unit == 0) & (52 < (now.second % 60) <= 57)) :   # N시:00:02초 ~ N시:00:07초 사이 시각이면
            
            DF_selected_coin = qualified_coin ()
            target_coin = DF_selected_coin.iloc[0]['coin']        
            check_coin_currency = target_coin[4:]   # 'KRW-' 이후의 문자열, 코인 종류에 따라 3자리수, 4자리수 등 다양함
            print ('target_coin : {0}  /  check_coin_currency : {1}'.format(target_coin, check_coin_currency))
        
            temp_current_price = get_current_price(target_coin)
            if temp_current_price >= 1000000 :  # 200만원 이상은 거래단위가 1000원, 100~200만원은 거래단위가 500원이지만 편의상 200만원 이상과 함께 처리
                unit_factor = -3
                unit_value = 1000
            elif temp_current_price >= 100000 :
                unit_factor = -2
                unit_value = 50
            elif temp_current_price >= 10000 :
                unit_factor = -1
                unit_value = 10
            elif temp_current_price >= 1000 :
                unit_factor = -1
                unit_value = 5
            elif temp_current_price >= 100 :
                unit_factor = 0
                unit_value = 1
            else :
                temp_current_price <= 100   # 100원 미만은 별도로 code에서 int형이 아닌 float형으로 형변환 해줘야함
                unit_factor = 1
                unit_value = 0.1
            
            
            opt_succesive_rise_No, opt_dec_acept_No = proper_rise_dec_condition (target_coin)   # 최적 opt_succesive_rise_No, opt_dec_acept_No 추출
            print ('opt_succesive_rise_No : {0}  /  opt_dec_acept_No : {1}'.format(opt_succesive_rise_No, opt_dec_acept_No))
        
            DF_last = pyupbit.get_ohlcv(target_coin, count = candle_count, interval = candle_adapt)
            DF_check = rise_tagging(DF_last, opt_succesive_rise_No)
            last_succesive_rise_No = DF_check['successive_rise'][-2]
            print ('last_succesive_rise_No : ', last_succesive_rise_No)
        
            not_buying_condition_check = not_buying_condition (DF_last, opt_succesive_rise_No)
            print ('not_buying_condition_check : ', not_buying_condition_check)
        
            if ((last_succesive_rise_No == opt_succesive_rise_No) & (not_buying_condition_check != 1)) :
                bought_price = temp_current_price + ((buy_margin -1) * unit_value)
                investable_budget = get_balance(check_currency) * invest_ratio
                bought_volume = (investable_budget * (1 - transaction_fee_ratio)) / bought_price
                transaction_buy = upbit.buy_limit_order(target_coin, bought_price, bought_volume)
                time.sleep(5)
                print ('buy_transaction_result :', transaction_buy)
                print ('bought_target_volume : {0}  /  bought_volume_until_now : {1}  /  now_in buying_check mode : {1}'.format(bought_volume, get_balance(check_coin_currency), (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600)))))
                buy_time = datetime.datetime.now().hour
                time.sleep(10)
            
                while ((datetime.datetime.now().minute % time_unit) < (time_unit -1)) :   # 한번에 매수 물량 전체가 매수가 안될것을 고려하여, 1 time unit 동안은 매수 시도 유지
                    print ('bought_target_volume : {0}  /  bought_volume_until_now : {1}  /  now_in buying_check mode : {1}'.format(bought_volume, get_balance(check_coin_currency), (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600)))))
                    #print ('bought_state : {0}  /  now_in buiyng_loop : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600)))))
                
                    if get_current_price(target_coin) <= bought_price * (1-sell_one_candle_force) :   # 만약 매수 시도 시간(1 time_unit) 중간에, 하락 수용 가능 수준 이상으로 하락하게 되면
                        transaction_sell = upbit.sell_market_order(target_coin, get_balance(check_coin_currency))   # 시장가에 매도
                        print ('\nnow :', (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor*3600))))
                        print ('sell_transaction_result :', transaction_sell)
                
                    time.sleep(1)
            
                print ('transaction_buy_cancel is considering')
                transaction_buy_cancel = upbit.cancel_order(transaction_buy['uuid'])   # 1시간 매수 시간 동안에도 매수가 미수에 그치면 매수 중단
    
        if get_balance(check_coin_currency) == 0 :
            bought_state = 0
            print ('bought_state is 0')
        
        if get_balance(check_coin_currency) > 0 :
            bought_state = 1
            print ('bought_state is 1')
 
    
        # 일반 매도 영역

        if (bought_state == 1) :
            if (now.minute % time_unit == 0) & (52 < (now.second % 60) <= 57) :
                DF_last = pyupbit.get_ohlcv(target_coin, count = candle_count, interval = candle_adapt)
                DF_check = rise_tagging(DF_last, opt_succesive_rise_No)
                last_succesive_rise_No = DF_check['successive_rise'][-2]
                print ('last_succesive_rise_No : ', last_succesive_rise_No)
            
                if last_succesive_rise_No == (opt_succesive_rise_No - opt_dec_acept_No) :   # 기준 수준 이상으로 하락이 연속되면
                    transaction_sell = upbit.sell_market_order(target_coin, get_balance(check_coin_currency))   # 시장가에 매도
                    print ('bought_state : {0}  / now_in selling_check mode : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600)))))
                    print ('sell_transaction_result :', transaction_sell)
                    time.sleep(10)
                
        # 특이 매도 영역
        # 1) 매수가 대비 하락폭이 일정 비율 보다 클때 강제 매도     
        if (bought_state == 1) :
            if (get_current_price(target_coin) <= bought_price * (1-sell_force)) :   # 하락폭이 기준수준보다 크다면
                transaction_sell = upbit.sell_market_order(target_coin, get_balance(check_coin_currency))   # 시장가에 매도
                print ('bought_state : {0}  / now_in FORCED selling_check mode : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600)))))
                print ('sell_transaction_result :', transaction_sell)
                time.sleep(10)
            
        # 2) 한 candle 안에서 하락폭이 해당 candle의 open 가격 대비 일정 비율 보다 클때 강제 매도
        if (bought_state == 1) :
            DF_one_candle_check = pyupbit.get_ohlcv(target_coin, count = 5, interval = candle_adapt)
            if (get_current_price(target_coin) <= ((1 - sell_one_candle_force) * DF_one_candle_check.iloc[-1]['open'])) :
                transaction_sell = upbit.sell_market_order(target_coin, get_balance(check_coin_currency))   # 시장가에 매도
                print ('bought_state : {0}  / now_in one_candle FORCED selling_check mode : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600)))))
                print ('sell_transaction_result :', transaction_sell)
                time.sleep(10)
            
        # 3) 매수가 대비 하락폭이 일정 비율 보다 클때 자동 매도       
        if (bought_state == 1) :
            if (get_current_price(target_coin) >= bought_price * (1 + sell_auto)) :   # 상승폭이 기준수준보다 크다면
                transaction_sell = upbit.sell_market_order(target_coin, get_balance(check_coin_currency))   # 시장가에 매도
                print ('bought_state : {0}  / now_in AUTO selling_check mode : {1}'.format(bought_state, (datetime.datetime.now() + datetime.timedelta(seconds = (time_factor * 3600)))))
                print ('sell_transaction_result :', transaction_sell)
                time.sleep(10)
    
        time.sleep(1)
    
    
    except Exception as e:
        print('ERROR')
        time.sleep(1)
'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




