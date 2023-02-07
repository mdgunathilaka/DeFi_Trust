import requests
from web3 import Web3, HTTPProvider
from web3.datastructures import AttributeDict
from hexbytes import HexBytes
import pandas as pd
from Crypto.Hash import keccak

INFURA_URL = ""
ABI = open("normal_token_abi.txt").read()

web3 = Web3(HTTPProvider(INFURA_URL))
print(web3.isConnected())


def get_rpc_response(method, list_params=[]):
    print(list_params)
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


def get_logs(contract, myevent, hash_create, from_block, to_block, number_batches):
    print("get_logs")
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
    block_list = [int(from_block + i * (to_block - from_block) / number_batches) for i in range(0, number_batches)] + [
        to_block]

    block_list[0] -= 1
    list_params = [[{"address": contract.address,
                     "fromBlock": hex(block_list[i - 1] + 1),
                     "toBlock": hex(block_list[i]),
                     "topics": [hash_create]}] for i in range(1, number_batches + 1)]

    logs = get_rpc_response("eth_getLogs", list_params)
    for j, log in enumerate(logs):
        if list(log.keys())[-1] == "result":
            for event in log['result']:
                log_dict = change_log_dict(event)
                events_clean += [clean_logs(contract, myevent, [log_dict])]
        else:
            events_clean += get_logs(contract, myevent, hash_create, int(list_params[j][0]["fromBlock"], 16),
                                     int(list_params[j][0]["toBlock"], 16), 15)
    return events_clean

def obtain_hash_event(event: str) -> str:
    k = keccak.new(digest_bits=256)
    k.update(bytes(event, encoding='utf-8'))
    return '0x'+k.hexdigest()

hash_log = obtain_hash_event('Transfer(address,address,uint256)')
hash_log2 = obtain_hash_event('Minted(address,uint256)')

def get_transfers(token_address, out_path, start_block, end_block, decimal=18):
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
    try:
        contract = web3.eth.contract(Web3.toChecksumAddress(token_address), abi=ABI)
        transfers = get_logs(contract, "Transfer", hash_log, start_block, end_block, number_batches=1)
        #transfers = get_logs(contract, "Transfer", hash_log2, start_block, end_block, number_batches=1)
    except Exception as err:
        print(f"Exception occured: {err}")
        return

    # Save txs in a Dataframe.
    txs = [[transaction['transactionHash'].hex(), transaction["blockNumber"], transaction["args"]['from'],
            transaction["args"]['to'], transaction["args"]['value'] / 10 ** decimal] for transaction in transfers]
    transfers = pd.DataFrame(txs, columns=["transactionHash", "block_number", "from", "to", "value"])
    transfers.to_csv(out_path + "/" + token_address + ".csv", index=False)
    return

#get_transfers('0x70cb2B8f546299bc26078473fD61ce80FC0f98e5', './', 12884209, 16233356)
#get_transfers('0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984', './', 10861674 , 16233751) #uni
#get_transfers('0x8DcFF4f1653f45cF418b0b3A5080A0fDCac577C8', '../event_logs', 11052662 , 16234209) #yai
#get_transfers('0x419D0d8BdD9aF5e606Ae2232ed285Aff190E711b', '../event_logs', 3988945 , 16238062) #fun
#get_transfers('0x9b6443b0fB9C241A7fdAC375595cEa13e6B7807A', '../event_logs', 4216651 , 16238062) #rcc
#get_transfers('0x9b6443b0fB9C241A7fdAC375595cEa13e6B7807A', '../test_series', 4216651 , 16238062) #rcc
get_transfers('0x8DcFF4f1653f45cF418b0b3A5080A0fDCac577C8', '../test_series', 11052662 , 16234209) #yai
#get_transfers('0x70cb2B8f546299bc26078473fD61ce80FC0f98e5', '../test_series', 12884209, 16233356)

