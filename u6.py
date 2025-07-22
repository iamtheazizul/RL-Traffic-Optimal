class SongNode:
	def __init__(self, artist, song, next=None):
		self.song = song
        self.artist = artist
		self.next = next

# For testing
def print_linked_list(node):
    current = node
    while current:
        print(current.song, end=" -> " if current.next else "")
        current = current.next
    print()
		
top_hits_2010s = SongNode("Uptown Funk", SongNode("Party Rock Anthem", SongNode("Bad Romance")))
# top_hits_2010s = SongNode()

print_linked_list(top_hits_2010s)
