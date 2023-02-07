import requests
from web3 import Web3, HTTPProvider
from web3.datastructures import AttributeDict
from hexbytes import HexBytes
import pandas as pd
from Crypto.Hash import keccak
from tqdm import tqdm

INFURA_URL = "INFURA_URL"
ABI = open("normal_token_abi.txt").read()

web3 = Web3(HTTPProvider(INFURA_URL))
print(web3.isConnected())


def get_rpc_response(method, list_params=[]):
    #print("get_rpc_response")
    #print(list_params)
    """
    Parameters
    ----------
    method: str
        Indicates node method.
    list_params: List[Dict[str, Any]]
        List of request parameters.

    Returns
    -------
    args_event: AttributeDict
        Change number basis.

    Example
    -------
        If we want token transfers of 0xa150Db9b1Fa65b44799d4dD949D922c0a33Ee606
        between blocks [11000000, 11025824] then:
        method: 'eth_getLogs'
        list_params: [[{'address': '0xa150Db9b1Fa65b44799d4dD949D922c0a33Ee606',
                    'fromBlock': '0xa7d8c0', 'toBlock': '0xa83da0',
                    'topics': ['0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef']}]]
    """
    url = INFURA_URL
    list_params = list_params or []
    data = [{"jsonrpc": "2.0", "method": method, "params": params, "id": 1} for params in list_params]
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=data)
    #print(response.json())
    return response.json()


def change_log_dict(log_dict):
    #print("change_logs")
    """
    Parameters
    ----------
    log_dict: AttributeDict
        Decoded logs.

    Returns
    -------
    args_event: AttributeDict
        Change number basis.
    """
    dictionary = log_dict.copy()
    dictionary['blockHash'] = HexBytes(dictionary['blockHash'])
    dictionary['blockNumber'] = int(dictionary['blockNumber'], 16)
    dictionary['logIndex'] = int(dictionary['logIndex'], 16)
    for i in range(len(dictionary['topics'])):
        dictionary['topics'][i] = HexBytes(dictionary['topics'][i])
    dictionary['transactionHash'] = HexBytes(dictionary['transactionHash'])
    dictionary['transactionIndex'] = int(dictionary['transactionIndex'], 16)
    return AttributeDict(dictionary)


def clean_logs(contract, myevent, log):
    #print("clean_logs")
    """
    Parameters
    ----------
    contract: web3.eth.contract
        Contract that contains the event.
    myevent: str
        string with event name.
    log: List[AttributeDict]
        List containing raw node response.

    Returns
    -------
    args_event: AttributeDict
        Decoded logs.
    """
    log_dict = AttributeDict({'logs': log})
    eval_string = 'contract.events.{}().processReceipt({})'.format(myevent, log_dict)
    args_event = eval(eval_string)[0]
    return args_event


def get_logs(contract, myevent, hash_create, from_block, block_range_=100):
    #print("get_logs")
    """
    Get event logs using recursion.

    Parameters
    ----------
    contract: web3.eth.contract
        Contract that contains the event.
    myevent: str
        string with event name.
    hash_create: str
        hash of the event.
    from_block: int
        Starting block.
    to_block: int
        Ending block.
    number_batches: int
        infura returns just 10k logs each call, therefore we need to split time series into batches.

    Returns
    -------
    events_clean: list
        List with all clean logs.
    """

    events_clean = []
    to_block = web3.eth.block_number
    head = from_block
    block_range = block_range_
    tail = head+block_range
    pbar = tqdm(total=to_block-from_block)

    while_b = True
    while (while_b):
        #print(f"scanning block range: {head} to {tail}")

        list_params = [[{"address": contract.address, "fromBlock": hex(head), "toBlock": hex(tail), "topics": [hash_create]}]]

        logs = get_rpc_response("eth_getLogs", list_params)

        #print("got logs rh")

        for j, log in enumerate(logs):
            #print(j, log)
            if list(log.keys())[-1] == "result":
                try:
                    for event in log['result']:
                        log_dict = change_log_dict(event)
                        events_clean += [clean_logs(contract, myevent, [log_dict])]
                        if len(events_clean) > 500:
                            return events_clean
                        #print(events_clean)
                    if(tail+block_range>to_block):
                        if(tail+block_range_>to_block):
                            while_b = False
                        else:
                            block_range = block_range_
                            head = tail
                            tail += block_range
                            pbar.update(block_range)
                    else:
                        head = tail
                        tail += block_range
                        block_range = block_range*2
                        pbar.update(block_range)
                except Exception as err:
                    print(f"Exception occured: {err}")
                    print(j, log)
            else:
                block_range = block_range// 2
                print(f"limit exceeded. new range: {block_range}")
                tail = head
                tail += block_range
    else:
        print(f"scanning block range: {head} to {to_block}")
        list_params = [[{"address": contract.address, "fromBlock": hex(head), "toBlock": hex(to_block), "topics": [hash_create]}]]
        logs = get_rpc_response("eth_getLogs", list_params)

        for j, log in enumerate(logs):
            if list(log.keys())[-1] == "result":
                for event in log['result']:
                    log_dict = change_log_dict(event)
                    events_clean += [clean_logs(contract, myevent, [log_dict])]
            else:
                print("limit exceeded")
    return events_clean

def obtain_hash_event(event: str) -> str:
    k = keccak.new(digest_bits=256)
    k.update(bytes(event, encoding='utf-8'))
    return '0x'+k.hexdigest()

hash_log = obtain_hash_event('Transfer(address,address,uint256)')
hash_log2 = obtain_hash_event('Minted(address,uint256)')

def get_transfers(token_address, out_path, start_block, decimal=18):
    """
    Get transfer logs for a given token and period.
    This function saves transfers as a csv in out_path.

    Parameters
    ----------
    token_address : str
        Token address.
    out_path : str
        Path to output directory.
    decimal: float
        Token decimal (usually 18).
    start_block: int
        Starting block.
    end_block: int
        Ending block.
    """

    # Initialise contract objects and get the transactions.
    contract = web3.eth.contract(Web3.toChecksumAddress(token_address), abi=ABI)
    transfers = get_logs(contract, "Transfer", hash_log, start_block)
    #transfers = get_logs(contract, "Transfer", hash_log2, start_block, end_block, number_batches=1)

    # Save txs in a Dataframe.
    txs = [[transaction['transactionHash'].hex(), transaction["blockNumber"], transaction["args"]['from'],
            transaction["args"]['to'], transaction["args"]['value'] / 10 ** decimal] for transaction in transfers]
    transfers = pd.DataFrame(txs, columns=["transactionHash", "block_number", "from", "to", "value"])
    transfers.to_csv(out_path + "/" + token_address + ".csv", index=False)
    return

import pandas as pd

healthy_df = pd.read_csv('healthy_tokens.csv')

w = 0
for i in healthy_df['token_address']:
    print(w)
    df = pd.read_csv('./healthy_event_series/'+i+'.csv')
    if len(df)< 50:
        print(i)
        get_transfers(str(i), './healthy_event_series', 1)
    w += 1