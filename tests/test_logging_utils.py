import dotenv

dotenv.load_dotenv(".env")

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def test_log():
    logging.info("test logging.info!")


if __name__ == "__main__":
    test_log()
