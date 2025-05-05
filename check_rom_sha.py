import retro
import hashlib


def get_rom_hash(rom_path):
    # Read the ROM file in binary mode
    with open(rom_path, "rb") as f:
        rom_data = f.read()
        # Calculate MD5 hash
        return hashlib.md5(rom_data).hexdigest().lower()


# Example usage:
hash1 = get_rom_hash("./rom.md")
print(f"ROM 1 Hash: {hash1}")

# Uncomment to test another ROM
# hash2 = get_rom_hash("./rom2.md")
# print(f"ROM 2 Hash: {hash2}")
