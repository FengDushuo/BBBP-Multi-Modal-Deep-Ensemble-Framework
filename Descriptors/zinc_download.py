import os
import sys
import requests
import pandas as pd
from colorama import Fore
from tqdm import tqdm
import concurrent.futures

# 下载单个分子数据
def download_molecule(zinc_id, zinc_version, zinc_file_type):
    url = f"https://zinc{zinc_version}.docking.org/substances/{zinc_id}.{zinc_file_type}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            content = response.text.strip()
            # 针对 smi 文件类型解析 SMILES 和 ZINC_ID
            if zinc_file_type == "smi":
                lines = content.splitlines()
                if len(lines) == 1 and " " in lines[0]:  # 格式为 "SMILES ZINC_ID"
                    smiles, zinc_id_returned = lines[0].split(" ", 1)
                    if zinc_id_returned == zinc_id:  # 确保返回的 ID 和请求一致
                        return {"ZINC_ID": zinc_id, "SMILES": smiles}
                    else:
                        print(Fore.RED + f"ID mismatch for {zinc_id}: {zinc_id_returned}" + Fore.RESET)
                        return None
                else:
                    print(Fore.RED + f"Unexpected format for {zinc_id}: {content}" + Fore.RESET)
                    return None
            else:
                # 对于非 SMILES 文件类型，保存原始内容
                return {"ZINC_ID": zinc_id, "Data": content}
        else:
            print(Fore.RED + f"Failed to download {zinc_id} (status code: {response.status_code})" + Fore.RESET)
            return None
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error downloading {zinc_id}: {e}" + Fore.RESET)
        return None

# 打开并验证 ZINC ID 列表
def list_opener(input_id_list):
    try:
        with open(input_id_list, 'r') as file:
            zinc_id_list = file.read().splitlines()
    except FileNotFoundError:
        print(Fore.RED + f"No such file {input_id_list} exists.")
        print(Fore.RESET + "Exiting...")
        sys.exit()

    valid_zinc_ids = [zinc_id.strip() for zinc_id in zinc_id_list if zinc_id.startswith("ZINC")]
    if not valid_zinc_ids:
        print(Fore.RED + "No valid zinc IDs found in the list.")
        print(Fore.RESET + "Exiting...")
        sys.exit()

    return valid_zinc_ids

# 主程序
def main():
    print(Fore.BLUE + "\n****************************************************")
    print("************* Zinc Molecule Downloader *************")
    print("****************************************************")
    print("****** Author: Modified for CSV generation ********")
    print("****************************************************\n")
    print(Fore.RESET + "Welcome.")

    zinc_version = input("Zinc version: (choose between 15 & 20)\n")
    while zinc_version not in ["15", "20"]:
        print(Fore.RED + "Invalid zinc version.")
        zinc_version = input(Fore.RESET + "Please choose a valid zinc version: (choose between 15 & 20)\n")

    zinc_file_type = input("Which file type to download? (choose between sdf, smi, csv, xml, json)\n")
    while zinc_file_type not in ["sdf", "smi", "csv", "xml", "json"]:
        print(Fore.RED + "Invalid file type.")
        zinc_file_type = input(Fore.RESET + "Please choose a valid file type: (choose between sdf, smi, csv, xml, json)\n")

    input_id_list = input("Zinc IDs list file: (default: list.txt; or to choose the default just press enter)\n") or "list.txt"
    zinc_id_list = list_opener(input_id_list)

    print(Fore.GREEN + f"Your chosen list contains {len(zinc_id_list)} molecules." + Fore.RESET)

    print("****************************************************\n")

    # 下载分子数据并保存到内存中
    results = []
    max_workers = os.cpu_count() * 2  # 调整线程数
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_molecule, zinc_id, zinc_version, zinc_file_type): zinc_id for zinc_id in zinc_id_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit="molecules"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(Fore.RED + f"Error: {e}" + Fore.RESET)

    print("****************************************************\n")
    print(Fore.GREEN + "Download job finished." + Fore.RESET)
    print("****************************************************\n")

    # 保存到 CSV 文件
    if results:
        df = pd.DataFrame(results)
        output_file = "zinc_dataset.csv"
        if zinc_file_type == "smi":
            df = df[["ZINC_ID", "SMILES"]]  # 确保只保存 ZINC_ID 和 SMILES
        df.to_csv(output_file, index=False)
        print(Fore.GREEN + f"Data saved to {output_file} with {len(results)} records." + Fore.RESET)
    else:
        print(Fore.RED + "No data downloaded successfully." + Fore.RESET)

if __name__ == '__main__':
    main()
