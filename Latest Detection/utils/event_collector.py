import requests
from web3 import Web3, HTTPProvider
from web3.datastructures import AttributeDict
from hexbytes import HexBytes
from Crypto.Hash import keccak
import pandas as pd

class ETH_API:
    def __init__(self):
        self.INFURA_URL = ""
        self.ABI = open("./utils/normal_token_abi.txt").read()
        self.web3 = Web3(HTTPProvider(self.INFURA_URL))
        self.hash_log = self.obtain_hash_event('Transfer(address,address,uint256)')
        print(f"Web Is conected: {self.web3.isConnected()}")

    def get_token_name(self, address):
        contract = self.web3.eth.contract(address, abi=self.ABI)
        token_name = contract.functions.name().call()
        token_symbol = contract.functions.symbol().call()
        return token_name, token_symbol

    def get_rpc_response(self, method, list_params=[]):
        # print("get_rpc_response")
        # print(list_params)
        url = self.INFURA_URL
        list_params = list_params or []
        data = [{"jsonrpc": "2.0", "method": method, "params": params, "id": 1} for params in list_params]
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data)
        # print(response.json())
        return response.json()

    def change_log_dict(self, log_dict):
        # print("change_logs")
        dictionary = log_dict.copy()
        dictionary['blockHash'] = HexBytes(dictionary['blockHash'])
        dictionary['blockNumber'] = int(dictionary['blockNumber'], 16)
        dictionary['logIndex'] = int(dictionary['logIndex'], 16)
        for i in range(len(dictionary['topics'])):
            dictionary['topics'][i] = HexBytes(dictionary['topics'][i])
        dictionary['transactionHash'] = HexBytes(dictionary['transactionHash'])
        dictionary['transactionIndex'] = int(dictionary['transactionIndex'], 16)
        return AttributeDict(dictionary)

    def clean_logs(self, contract, myevent, log):
        # print("clean_logs")
        log_dict = AttributeDict({'logs': log})
        eval_string = 'contract.events.{}().processReceipt({})'.format(myevent, log_dict)
        args_event = eval(eval_string)[0]
        return args_event

    def get_logs(self, contract, myevent, hash_create, block_range=10000):
        # print("get_logs")
        events_clean = []
        to_block = self.web3.eth.block_number
        head = to_block - block_range
        tail = to_block

        while_b = True
        while (while_b):
            #print(f"scanning block range: {head} to {tail}")

            list_params = [
                [{"address": contract.address, "fromBlock": hex(head), "toBlock": hex(tail), "topics": [hash_create]}]]

            logs = self.get_rpc_response("eth_getLogs", list_params)

            # print("got logs rh")

            for j, log in enumerate(logs):
                # print(j, log)
                if list(log.keys())[-1] == "result":
                    try:
                        part = []
                        for event in log['result']:
                            log_dict = self.change_log_dict(event)
                            #events_clean = [self.clean_logs(contract, myevent, [log_dict])] + events_clean
                            part += [self.clean_logs(contract, myevent, [log_dict])]
                            # print(events_clean)
                        events_clean = part + events_clean
                        if (len(log['result']) < 5000):
                            block_range = block_range * 2
                        tail = head
                        head = tail - block_range

                    except Exception as err:
                        print(f"Exception occured: {err}")
                        print(j, log)
                else:
                    print(log)
                    block_range = block_range // 2
                    print(f"limit exceeded. new range: {block_range}")
                    head = tail - block_range
            if len(events_clean) > 1080:
                print(f"events_full")
                while_b = False
            #May-04-2020 04:34:02 PM +UTC - Uniswap V2 deployment
            if (tail<10000835):
                print(f"block_range: {block_range} tail: {tail}")
                while_b = False
        if len(events_clean)>1080:
            events_clean = events_clean[len(events_clean)-1080:]
        return events_clean

    def obtain_hash_event(self, event: str) -> str:
        k = keccak.new(digest_bits=256)
        k.update(bytes(event, encoding='utf-8'))
        return '0x' + k.hexdigest()

    def get_transfers(self, token_address, out_path, decimal=18):
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
        contract = self.web3.eth.contract(Web3.toChecksumAddress(token_address), abi=self.ABI)
        transfers = self.get_logs(contract, "Transfer", self.hash_log)
        # transfers = get_logs(contract, "Transfer", hash_log2, start_block, end_block, number_batches=1)

        # Save txs in a Dataframe.
        txs = [[transaction['transactionHash'].hex(), transaction["blockNumber"], transaction["args"]['from'],
                transaction["args"]['to'], transaction["args"]['value'] / 10 ** decimal] for transaction in transfers]
        transfers = pd.DataFrame(txs, columns=["transactionHash", "block_number", "from", "to", "value"])
        transfers.to_csv(out_path + "/" + token_address + ".csv", index=False)
        return transfers
