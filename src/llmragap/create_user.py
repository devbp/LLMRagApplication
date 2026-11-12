import argparse
import getpass

from llmragap.auth_api import USERS_TABLE, get_supabase, hash_password


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a login user to Supabase.")
    parser.add_argument("username")
    args = parser.parse_args()

    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords do not match.")
    if not password or len(password.encode()) > 72:
        raise SystemExit("Password must be 1 to 72 bytes long.")

    get_supabase().table(USERS_TABLE).insert(
        {"username": args.username, "password_hash": hash_password(password)}
    ).execute()
    print(f"Created user {args.username}.")


if __name__ == "__main__":
    main()
