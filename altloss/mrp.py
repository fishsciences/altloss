import cmdstanpy as stan
import pandas as pd
import numpy as np
import numpy.random as rnd
import itertools as it
import sys


# mapping from hydrophones to states
# necessary because the names of the hydrophones 
# is not uniform throughout the data 
_state_map = {
    'RGU': {'RGU', 'RGUP', 'RG_UP'}, 
    'RGD': {'RGD', 'RGDOWN', 'RG_DOWN'}, 
    'BAY': set(), # the forebay is not directly observable
    'IC':  {'IC1', 'IC1_FISHERMANS POINT', "IC1 FISHERMAN'S PT", 'IC2'},
    'LVR': {'IC3', 'IC3_VR2W', 'IC3 TRASH RACK', 'P', 'P1', 'P5', 'P4_A', 'P4_B'}, 
    'HLD': {'S', 'S1_UP', 'S1_DOWN', 'S2_UP_LEFT', 'S2_UP_RIGHT', 'S2_DOWN'}.union({f'HTB{n}' for n in range(10)}),
    'FSB': {'FSB', 'FSB_A', 'FSB_B', 'FISH SCIENCE BUILDING'},
    'SVG': {'CLRS_A', 'CLRS_B', 'HBRS_A', 'HBRS_B', 'HORSESHOE BEND', 'CURTIS LANDING DOWN', 'RELEASESITE'},
    'DTH': set(), # death is not directly observable
    'OUT': set() # can only be observed indirectly
}

# all the allowed transitions
_next_map = {
    'RGU': {'RGD', 'OUT'},
    'RGD': {'RGU', 'BAY', 'DTH'},  
    'BAY': {'RGD', 'IC', 'DTH'},
    'IC':  {'BAY', 'LVR', 'DTH'},
    'LVR': {'FSB', 'IC', 'HLD', 'DTH'},
    'HLD': {'LVR', 'SVG', 'DTH'},
    'FSB': {'LVR', 'DTH'},
    'SVG': set(), # absorbing state
    'DTH': set(), # absorbing state
    'OUT': set()  # absorbing state
}

_from_map = {
    'RGU': {'RGD', 'BAY', 'IC',  'LVR', 'HLD', 'FSB'},
    'RGD': {'RGU', 'BAY', 'IC',  'LVR', 'HLD', 'FSB'},
    'BAY': {'RGD', 'RGU', 'IC',  'LVR', 'HLD', 'FSB'},
    'IC':  {'RGD', 'RGU', 'BAY', 'LVR', 'HLD', 'FSB'},
    'LVR': {'RGD', 'RGU', 'BAY', 'IC',  'HLD', 'FSB'},
    'HLD': {'RGD', 'RGU', 'BAY', 'IC',  'LVR', 'FSB'},
    'FSB': {'RGD', 'RGU', 'BAY', 'IC',  'LVR', 'HLD'},
    'SVG': {'RGD', 'RGU', 'BAY', 'IC',  'LVR', 'FSB', 'HLD'},
    'DTH': {'RGD', 'BAY', 'IC',  'LVR', 'FSB', 'HLD'},
    'OUT': {'RGD', 'RGU', 'BAY', 'IC', 'LVR', 'FSB', 'HLD'}
}


# Given a one-to-many mapping from states to hydrophones,
# this function computes a one-to-one mapping from
# hydrophones to states. If a hydrophone maps to multiple
# states, an exception is raised
def invert_state_map(state_map):
    station_map = dict()
    for (k, s) in _state_map.items():
        for v in s:
            if v in station_map:
                raise ValueError(f'station {v} maps to multiple states')
            station_map[v] = k
    return station_map

_station_map = invert_state_map(_state_map)


def assign_state(df, time_col='UTC8'):
    # covert time stamps to Pandas datetime, and sort the data
    # by fish (transmitter) and time.
    # df['TransmitterName'] = df['Transmitter'].map(lambda x: x.rsplit('-')[-1])
    df['TransmitterName'] = df['Transmitter']
    df = df.rename(columns={time_col: 'Time'})
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values(by=['Transmitter', 'TransmitterName', 'Time'])
    df = df.groupby('Transmitter').apply(battery_trim).reset_index(drop=True)
    # compute the state the fish is in, 
    # and drop rows where the state could
    # not be determined
    df = df.assign(**{'State': df['StationName'].map(lambda x: _station_map.get(x.strip().upper()))})
    df = df.dropna(subset=['State'])
    # group by fish and compute the state and time stamp
    # of the previous and next detections
    grp = df.groupby(['Transmitter'])
    df['PrevState'] = grp['State'].shift(1)
    df['PrevTime'] = grp['Time'].shift(1)
    df['NextState'] = grp['State'].shift(-1)
    df['NextTime'] = grp['Time'].shift(-1)
    # if the state has changed since the last detection
    # then this is an entry detection, i.e. the earliest
    # detection in a new state
    df['Entry'] = (df['State'] != df['PrevState'])
    # if the state changes in the next detection
    # then this is an exit observation, i.e. the latest
    # time the fish was observed in the current state
    df['Exit'] = (df['State'] != df['NextState'])
    # we only need the entry/exit detections
    df = df[df['Entry'] | df['Exit']]
    # Assign serial numbers to observations by
    # counting the entry observations 
    grp = df.groupby(['Transmitter'])
    df['Obs'] = grp['Entry'].cumsum()
    return df


# trim observations from one tag to battery life limit
def battery_trim(df, dt=pd.to_timedelta('180d')):
    mask = (df['Time'] - df['Time'].min()) <= dt
    return df.loc[mask, :]


# this function adds implicit observations to the data
# for example, a fish could be observed at RGD and then
# at IC1, but this canot happen without going through
# BAY, so we need to add a RGT->BAY transition and a 
# BAY -> IC1 transition
def add_intermediate_obs(df, src, dst, inter):
    # src -> inter -> dst transitions
    entry_mask = (df['State'] == src) & (df['NextState'] == dst) & df['Exit']
    exit_mask = (df['State'] == dst) & (df['PrevState'] == src) & df['Entry']
    # new entries
    df_entry = df[entry_mask].copy()
    df_entry['State'] = inter
    df_entry['PrevState'] = src
    df_entry['NextState'] = inter
    df_entry['PrevTime'] = df_entry['Time']
    df_entry['Exit'] = False
    df_entry['Entry'] = True
    # this a bit of a kludge, but it does its intended job:
    # it provides an observation id that fits between the 
    # consecutive observation ids of the existing `src` and
    # `dst` observations
    df_entry['Obs'] += 0.5 
    # new exits
    df_exit = df[exit_mask].copy()
    df_exit['State'] = inter
    df_exit['PrevState'] = inter
    df_exit['NextState'] = dst
    df_exit['NextTime'] = df_exit['Time']
    df_exit['Exit'] = True
    df_exit['Entry'] = False
    df_exit['Obs'] -= 0.5 
    # old exits
    df.loc[entry_mask, 'NextState'] = inter
    df.loc[entry_mask, 'NextTime'] = df.loc[entry_mask, 'Time']
    # old entries
    df.loc[exit_mask, 'PrevState'] = inter
    df.loc[exit_mask, 'PrevTime'] = df.loc[exit_mask, 'Time']
    res = pd.concat([df, df_entry, df_exit])
    res = res.sort_values(by=['Transmitter', 'Time', 'Obs'])
    return res

def add_bay_obs(df):
    df = add_intermediate_obs(df, 'RGD', 'IC', 'BAY')
    df = add_intermediate_obs(df, 'IC', 'RGD', 'BAY')
    return df

def add_louver_obs(df):
    df = add_intermediate_obs(df, 'LVR', 'SVG', 'HLD')
    df = add_intermediate_obs(df, 'HLD', 'IC', 'LVR')
    df = add_intermediate_obs(df, 'IC', 'HLD', 'LVR')
    df = add_intermediate_obs(df, 'HLD', 'FSB', 'LVR')
    df = add_intermediate_obs(df, 'FSB', 'HLD', 'LVR')
    return df

def add_final_obs(df):
    # RGU -> OUT
    mask = df['NextState'].isna() & (df['State'] == 'RGU')
    df.loc[mask, 'NextState'] = 'OUT'
    # DTH
    mask = df['NextState'].isna()
    df.loc[mask, 'NextState'] = 'DTH'
    return df

# the death state and the forebay state are not directly
# observable, so we add inferred observations for these
# states
def add_synthetic_obs(df):
    df = add_bay_obs(df)
    df = add_louver_obs(df)
    df = add_final_obs(df)
    return df

# this function computes upper and lower bounds
# for the residence time and calculates the set 
# of possible next states for a single observation
# (when the next state is not directly observed);
# 
# the intended use is that the pre-processed
# observation data is groupped by fish and 
# observation serial number, and then this function
# is applied to each group
def extract_one_obs(df, min_dt=pd.to_timedelta('60s')):
    state = df.iloc[0]['State']
    prev_state = df.iloc[0]['PrevState']
    next_state = df.iloc[-1]['NextState']
    prev_time = df.iloc[0]['PrevTime']
    next_time = df.iloc[-1]['NextTime']
    entry_time = df.iloc[0]['Time']
    exit_time = df.iloc[-1]['Time']
    if pd.isna(prev_time):
        entry_low = entry_time
        entry_high = entry_time
    else:
        entry_low = prev_time
        entry_high = entry_time
    if pd.isna(next_time):
        exit_low = exit_time
        exit_high = exit_time
    else:
        exit_low = exit_time
        exit_high = next_time
    # the shortest possible residence time  
    dt_low = exit_low - entry_high
    # the longest possible residence time
    dt_high = exit_high - entry_low
    # the uncertanty in residence time cannot 
    # be shorter than the time interval between 
    # pings
    if dt_high - dt_low < min_dt:
        dt_high = dt_low + min_dt 
    # convert waiting times to seconds
    dt_low = dt_low.total_seconds()
    dt_high = dt_high.total_seconds()
    if next_state in _next_map[state]:
        # if the state the fish transitioned to 
        # was directly observed, then the set of
        # possible next states is a singleton
        states = {next_state}
    else:
        # otherwise, 
        states = _from_map[next_state].intersection(_next_map[state])
    if len(states) < 1:
        print(f'No transitions possible: current = {state}, next = {next_state}', file=sys.stderr) 
    res = {
        'State': [state],
        'NextState': [states],
        'T_lower': [dt_low],
        'T_upper': [dt_high]
    }
    return pd.DataFrame(data=res)

# this function puts it all toghether: it segments data
# into observations, calculates entries and exits, adds
# implicit observations, calculates lower and upper bounds
# for residence times, and generates the set of possible
# next states for the transitions where the next sate is
# not directly observed
def process_obs_data(df, setup=None):
    df = assign_state(df)
    df = add_synthetic_obs(df)
    df['Obs'] = np.round(10*df['Obs']).astype(int)
    grp = df.groupby(['Transmitter', 'TransmitterName', 'Obs'])
    df = grp.apply(extract_one_obs)
    df = df.loc[df['NextState'].apply(lambda x: len(x) > 0)]
    if not (setup is None):
        df['Setup'] = setup
    df = df.reset_index().drop(columns=['level_3', 'Obs'])
    return df

# write out processed observation data
def write_obs_data(df, output='output.csv'):
    df.to_csv(output, index=False, header=True)

def read_raw_data(setups):
    dfs = ((setup, pd.read_csv(csv)) for (setup, csv) in setups.items())
    dfs = (process_obs_data(df, setup=s) for (s, df) in dfs)
    df = pd.concat(dfs)
    return df

# correctly read in data written by `write_obs_data`
def read_obs_data(csv):
    convs = {
        'State': str,
        'NextState': eval,
        'Transmitter': str,
        'TransmitterName': str,
        'Setup': str,
        'T_lower': float,
        'T_upper': float
    }
    df = pd.read_csv(csv, converters=convs)
    return df


##### Utility functions for reading and normalizing tag data #####

_normalize_tagging_columns_map = {
    # RFID     
    'RFID_Number': 'RFID', 
    'RFIDNumber': 'RFID',
    'RFID Number': 'RFID',
    # Acoustic tag
    'AcousticTag': 'AcousticTag', 
    'Acoustic_Tag': 'AcousticTag', 
    'Acoustic_Tag_ID': 'AcousticTag',
    # Weight
    'Weight': 'Weight', 
    'Weight (g)': 'Weight', 
    'Weight(g)': 'Weight',
    # Fork length
    'FL': 'FL', 
    'FL (mm)': 'FL', 
    'FL(mm)': 'FL',
    # Tag type
    'TagType': 'TagType', 
    'Tag_Type': 'TagType',
    # Fish species
    'Species': 'Species'
}

def normalize_tagging_cols(df, setup=None):
    df = df.rename(columns=_normalize_tagging_columns_map)
    df = df[[x for x in set(_normalize_tagging_columns_map.values())]]
    if not (setup is None):
        df['Setup'] = setup
    return df

def read_raw_tags(setups):
    dfs = ((setup, pd.read_csv(csv,)) for (setup, csv) in setups.items())
    dfs = (normalize_tagging_cols(df, setup=s) for (s, df) in dfs)
    df = pd.concat(dfs)
    df = df.dropna()
    df = df.loc[df['TagType'].map(str.upper) == 'ACOUSTIC']
    df.loc[df['FL'] < 75, 'FL'] = 10 * df.loc[df['FL'] < 75, 'FL']
    df['AcousticTag'] = df['AcousticTag'].map(lambda x: str(int(round(x))))
    return df

def read_tag_data(csv):
    convs = {
        'RFID': str,
        'AcousticTag': str,
        'TagType': str,
        'Setup': str,
        'Species': str,
        'FL': float,
        'Weight': float
    }
    df = pd.read_csv(csv, converters=convs)
    return df

##################################################################
# functions for reading in preprocessed observation data and     #
# computing the inputs expected by the Stan model                #
##################################################################

# for inference, we have to map states to a range of consecutive
# integers 1, 2, ..., N, since the Stan model uses the state
# to index various arrays
_stan_state_map = dict((k, i + 1) for (i, k) in enumerate(_state_map.keys()))


# binary encoding of possible next states for each observation
def _next_vect(next_set, state_map):
    res = np.zeros(len(state_map), dtype=np.float64)
    ix = [state_map[x] for x in next_set]
    ix = np.array(ix, dtype=int)
    res[ix - 1] = 1
    return res

# produce input data for the Stan model from acoustic observations
# of a given species.
def load_stan_data(obs_file, tag_file, species, state_map=_stan_state_map):
    tags = read_tag_data(tag_file)
    obs = read_obs_data(obs_file)
    # replace state names with state indices in observations
    obs['State'] = obs['State'].map(lambda x: state_map[x])
    obs['NextState'] = obs['NextState'].map(lambda x: _next_vect(x, state_map))
    # select only fish of the right species
    tags = tags.loc[tags['Species'].isin(species)]
    tags = tags.reset_index(drop=True).reset_index()
    tags = tags.rename(columns={'index': 'Fish'})
    tags['Fish'] = tags['Fish'] + 1
    # add fish indices to observations
    obs = obs.merge(
        tags.loc[:, ['AcousticTag', 'Fish']], 
        left_on='TransmitterName',
        right_on='AcousticTag',
        how='inner'
    )
    # start populating Stan data
    data = dict()
    # fish data
    data['N_fish'] = len(tags)
    data['fl_data'] = tags['FL'].values
    # observation data
    data['N_obs'] = len(obs)
    data['fish'] = obs['Fish'].values
    data['curr'] = obs['State'].values
    next_st = obs['NextState'].values.tolist()
    next_st = [x.tolist() for x in next_st]
    data['next'] = np.array(next_st, dtype=np.int64);
    data['t_lower'] = obs['T_lower'].values
    data['t_upper'] = obs['T_upper'].values
    # transition structure
    data['N_states'] = len(_stan_state_map)
    data['N_transient'] = data['N_states'] - 3
    trans = [
        [
            [_stan_state_map[s1], _stan_state_map[s2]] 
            for s2 in adj
        ] 
        for (s1, adj) in _next_map.items()
    ]
    trans = np.array(sum(trans, start=[]), dtype=int)
    data['transitions'] = trans.T
    data['N_transitions'] = trans.shape[0]
    return data


#####################################################
#                                                   #
#     Simulation functions                          #
#                                                   #
#####################################################


def draw_batch(data, fit, start='RGD', offset=0, size=1000):
    # data = input used for Stan model
    # fit  = output posterior sample from Stan model
    N_smp = fit.chains * fit.num_draws_sampling
    N_fish = data['N_fish']
    N_states = data['N_states']
    N_trans = data['N_transitions']
    # index transitions
    tmp = np.full((N_states, N_states), fill_value=N_trans+2)
    trans = data['transitions'] - 1 # Python indexes from 0, while Stan indexes from 1
    tmp[trans[0, :], trans[1, :]] = np.arange(N_trans, dtype=int)
    trans = tmp
    # re-sample, with replacement, from Stan output 
    smp = rnd.choice(N_smp, size=size, replace=True)
    fish = rnd.choice(N_fish, size=size, replace=True)
    var = fit.stan_variables() 
    lambda_res = var['lambda_res'][smp, fish, :]
    kappa_res = var['kappa_res'][smp, :]
    P = np.exp(var['logP'][smp, :, :])
    P[P < np.finfo(np.float64).eps] = 0.0
    P = P / P.sum(axis=2)[:, :, None]
    P = P.cumsum(axis=2)
    state = np.full(size, _stan_state_map[start] - 1, dtype=int)
    time = np.zeros(size, dtype=np.float64)
    active = np.arange(size, dtype=int)
    batch = {
        'offset': offset,
        'size': size, 
        'transient': data['N_transient'],
        'transitions': trans,
        'P': P,
        'lambda': lambda_res,
        'kappa': kappa_res,
        'state': state,
        'time': time,
        'active': active
    }
    return batch

# simulate a single transition evolution of a
# batch of synthetic fish, according to the Markov 
# renewal process
def advance_batch(batch):
    ix = batch['active']
    state = batch['state'][ix]
    t = batch['time'][ix]
    p = batch['P'][ix, state, :]
    dice = rnd.uniform(0, 1, size=len(ix))
    next_state = (p > dice[:, None]).argmax(axis=1)
    trans = batch['transitions'][state, next_state]
    lambda_res = batch['lambda'][ix, trans]
    kappa_res = batch['kappa'][ix, trans]
    dt = lambda_res * rnd.weibull(kappa_res)
    batch['state'][ix] = next_state
    batch['time'][ix] = t + dt
    return batch
    
# produce a snapshot in time for a batch
# of simulated fish
def batch_to_dataframe(batch):
    ix = batch['active']
    fish = batch['offset'] + ix;
    time = batch['time'][ix].copy()
    state = batch['state'][ix].copy()
    df = pd.DataFrame(data={'Fish': fish, 'State': state, 'Time': time})
    return df

# mark the fish that have reached an absorbing 
# state as inactive
def prune_batch(batch):
    ix = batch['active']
    state = batch['state'][ix]
    jx = ix[state < batch['transient']]
    batch['active'] = jx
    return batch

# full simulation of a batch of synthetic fish
def iter_batch(batch):
    while len(batch['active'] > 0):
        # if any fish are still active, we 
        # output a new snapshot, prune the fish 
        # that have reached absorbing states, and
        # simulate a new transition for each fish
        yield batch_to_dataframe(batch)
        batch = prune_batch(batch)
        batch = advance_batch(batch)

# this function produces several batches of given size
# its use in simulation is recommended for computational 
# speed: simulating several small batches that fit in the
# CPU cache is faster than simulating one large batch
def run_batches(data, fit, batch=1000, reps=1000):
    batches = (batch * r for r in range(reps))
    batches = (draw_batch(data, fit, offset=b, size=batch) for b in batches)
    batches = (iter_batch(b) for b in batches)
    batches = it.chain.from_iterable(batches)
    df = pd.concat(batches)
    # rescale time
    time_scale = 0.5 * (data['t_lower'] + data['t_upper']).mean()
    df['Time'] = df['Time'] * time_scale
    # translate to huma readable states
    states = sorted(_stan_state_map.items(), key=(lambda x: x[1]))
    states = [s for (s, _) in states]
    df['State'] = df['State'].apply(lambda k: states[k])
    return df
