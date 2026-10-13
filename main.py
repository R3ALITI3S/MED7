
def main():
    print("Starting preprocessing pipeline...")
    df = load_and_preprocess_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    print("Preprocessing completed successfully!")


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
