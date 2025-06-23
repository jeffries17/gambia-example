import pyperclip

# Get text from clipboard
text = pyperclip.paste()

# Process links
links = [line.strip() for line in text.splitlines() if line.strip()]
clean_links = [link.split('#')[0] for link in links]
hotel_links = [link for link in clean_links if 'Hotel_Review' in link]
unique_links = list(dict.fromkeys(hotel_links))

# Copy unique links back to clipboard
pyperclip.copy('\n'.join(unique_links))

# Print summary
print(f"Hotel Links Cleaner:")
print(f"Original links: {len(links)}")
print(f"Hotel links found: {len(hotel_links)}")
print(f"Unique hotel links: {len(unique_links)}")
print(f"Result copied to clipboard!")

# Optionally save to file
save_option = input("\nSave to file? (y/n): ").strip().lower()
if save_option == 'y':
    with open('clean_hotel_links.txt', 'w', encoding='utf-8') as f:
        for link in unique_links:
            f.write(f"{link}\n")
    print(f"Saved {len(unique_links)} links to clean_hotel_links.txt")