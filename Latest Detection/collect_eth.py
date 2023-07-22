import requests
from web3 import Web3, HTTPProvider
from web3.datastructures import AttributeDict
from hexbytes import HexBytes
from Crypto.Hash import keccak
import pandas as pd
import os
from datetime import datetime
from requests import get

class ETH_API:
    def __init__(self):
        self.INFURA_URL = "YOUR_INFURA_API_KEY"
        self.ABI = open("normal_token_abi.txt").read()
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

    def get_logs(self, contract, myevent, hash_create, block_range_=10000):
        # print("get_logs")
        events_clean = []
        to_block = self.web3.eth.block_number
        block_range = block_range_
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
                            #events_clean += [self.clean_logs(contract, myevent, [log_dict])]
                            part += [self.clean_logs(contract, myevent, [log_dict])]
                            # print(events_clean)
                        events_clean = part + events_clean
                        if (len(log['result']) < 100):
                            block_range = block_range * 2
                        tail = head
                        head = tail - block_range

                    except Exception as err:
                        print(f"Exception occured: {err}")
                        print(j, log)
                else:
                    #print(log)
                    block_range = block_range // 2
                    #print(f"limit exceeded. new range: {block_range}")
                    head = tail - block_range
            if len(events_clean) > 1080:
                #print(len(events_clean))
                print(f"events_full")
                while_b = False
            # May-04-2020 04:34:02 PM +UTC - Uniswap V2 deployment
            if (head < 10000835):
                if tail - block_range_ < 10000835:
                    print(f"block_range: {block_range} tail: {tail}")
                    while_b = False
                else:
                    block_range = block_range_
                    head = tail - block_range
        if len(events_clean)>1080:
            events_clean = events_clean[len(events_clean)-1080:]
        #print(len(events_clean))
        return events_clean, tail

    def obtain_hash_event(self, event: str) -> str:
        k = keccak.new(digest_bits=256)
        k.update(bytes(event, encoding='utf-8'))
        return '0x' + k.hexdigest()

    def collect_gas_time(self, token_address, start_block):
        API_KEY = os.environ["YOUR_ETHERSCAN_TOKEN"] #YOUR ETHERSCAN API KEY
        last_block = self.web3.eth.block_number
        txhashes = []
        timestamps = []
        gases = []
        pg = 1

        while (True):
            url = "https://api.etherscan.io/api?module=logs&action=getLogs&fromBlock=" + str(start_block) + "&toBlock=" + str(
                last_block) + "&address="+token_address+"&topic0=" + self.hash_log + "&page=" + str(
                pg) + "&offset=1000&apikey=" + API_KEY
            response = get(url)
            log = response.json()
            #print(f"pg:{pg}")
            #print(log)
            if log['status'] == '1':
                for event in log['result']:
                    # print(event)
                    log_dict = AttributeDict(event)
                    timestamp_string = int(log_dict['timeStamp'], 16)
                    #print(log_dict['transactionHash'])
                    try:
                        gas_price = int(log_dict['gasPrice'], 16) / 10 ** 18  # In wei
                    except:
                        gas_price = 0
                    timestamp = datetime.fromtimestamp(timestamp_string)
                    timestamps.append(str(timestamp))
                    gases.append(gas_price)
                    txhashes.append(log_dict['transactionHash'])
            else:
                print("gas done")
                #print(log)
                break
            pg += 1
        time_gas = pd.DataFrame(
            {'transactionHash': txhashes,
             'timestamp': timestamps,
             'gas_price': gases
             })
        return time_gas
        #return timestamps, gases

    def get_transfers(self, token_address, decimal=18):
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
            contract = self.web3.eth.contract(Web3.toChecksumAddress(token_address), abi=self.ABI)
        except:
            return
        transfers, first_block = self.get_logs(contract, "Transfer", self.hash_log)
        # transfers = get_logs(contract, "Transfer", hash_log2, start_block, end_block, number_batches=1)

        # Save txs in a Dataframe.
        txs = [[transaction['transactionHash'].hex(), transaction["blockNumber"], transaction["args"]['from'],
                transaction["args"]['to'], transaction["args"]['value'] / 10 ** decimal] for transaction in transfers]
        #transfers = pd.DataFrame(txs, columns=["transactionHash", "block_number", "from", "to", "value"])
        time_gas = self.collect_gas_time(token_address, first_block)
        #hashes, times, gases = self.collect_gas_time(token_address)
        #transfers.to_csv(out_path + "/" + token_address + "tfs.csv", index=False)
        #time_gas.to_csv(out_path + "/" + token_address + "tgs.csv", index=False)

        #transfers = transfers.merge(time_gas, on='transactionHash', how='left')
        '''
        if len(txs)!=len(times):
            print(f"len wrong {len(txs)} trasnactions, {len(times)} timestamps")
            return
        '''
        ltx = len(txs)
        print(f"ltx {ltx}")
        lts = len(time_gas)
        print(f"lts {lts}")
        j=0
        gas_tot = []
        for i in range(ltx):
            while(j<lts):
                if time_gas['transactionHash'][j]==txs[i][0]:
                    txs[i].append(time_gas['timestamp'][j])
                    txs[i].append(time_gas['gas_price'][j])
                    gas_tot.append(time_gas['gas_price'][j])
                    j+=1
                    break
                j+=1
            #txs[i].append(time_gas[i])
            #txs[i].append(gases[i])

        transfers = pd.DataFrame(txs, columns=["transactionHash", "block_number", "from", "to", "value", "timestamp", "gas_price"])
        #print(f"before {len(transfers)}")
        #transfers = transfers.dropna()
        transfers = transfers.fillna(sum(gas_tot)/len(gas_tot))
        #print(f"after {len(transfers)}")

        #transfers.to_csv(out_path + "/" + token_address + ".csv", index=False)
        return transfers

