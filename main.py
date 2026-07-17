def main():
    print("=" * 50)
    print("AI Banking Fraud Detection System")
    print("=" * 50)

if __name__ == "__main__":
    main()

    from src.preprocessing.data_loader import load_data
    from src.preprocessing.data_loader import load_data
from src.preprocessing.clean_data import clean_data
from src.preprocessing.eda import perform_eda


df = load_data("data/raw/bank_transactions.csv")

print(df.head())

df = load_data("data/raw/bank_transactions.csv")

df = clean_data(df)

perform_eda(df)

