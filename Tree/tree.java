// Trie data structure
class TrieNode {
	// R links to node children
	private TrieNode[] links;

	private final int R = 26;

	private boolean isEnd;

	public TrieNode(){
		links = new TrieNode[R];
	}

	public boolean containsKey(char ch){
		return links[ch - 'a'] != null;
	}

	public TrieNode get(char ch){
		return links[ch - 'a'];
	}

	public void put(char ch, TrieNode node){
		links[ch - 'a'] = node;
	}

	public void setEnd(){
		isEnd = true;
	}

	public boolean isEnd(){
		return isEnd;
	}
}

class Trie {
	private TrieNode root;

	public Trie(){
		root = new TrieNode();
	}

	// Inserts a word into the trie
	// Time: O(m), where m is the key length
	// Space: O(m), In the worst case newly inserted key doesn't share a prefix with the
	// keys already inserted in the trie. We have to add m new nodes, which takes us O(m)
	// space
	public void insert(String word){
		TrieNode node = root;
		for (int i = 0; i < word.length(); i++){
			char ch = word.charAt(i);
			if(!node.containsKey(ch)){
				node.put(ch, new TrieNode());
			}
			node = node.get(ch)
		}
		node.setEnd();
	}

	// Search a prefix or whole key in trie and
	// returns the node where search ends
	// Time: O(m) In each step of the algorithm we search for the next key
	// character.  In the worst case the algorithm performs m operations.
	// Space: O(1)
	public boolean search(String word){
		TrieNode node = root;
		for (int i = 0; i < word.length(); i++){
			char ch = word.charAt(i);
			if(!node.containsKey(chr)){
				return false;
			}
			node = node.get(ch);
		}
		return node != null && node.isEnd();
	}

	// Returns if there is any word in the trie
	// that starts with the given prefix
	public boolean startsWith(String prefix){
		TrieNode node = root;
		for (int i = 0; i < word.length(); i++){
			char ch = word.charAt(i);
			if(!node.containsKey(chr)){
				return false;
			}
			node = node.get(ch);
		}
		return node != null;
	}


}