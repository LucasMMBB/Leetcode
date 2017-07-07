public class ReverseLinkedList{
	public static void main (String[] args) throws java.lang.Exception{
		LinkedListT a = new LinkedListT();
		a.addAtBegin(5);
		a.addAtBegin(10);
		a.addAtBegin(15);
		a.addAtBegin(20);
	}
}

class Node{
	public int data;
	public Node next;
	public Node(int data){
		this.val = data;
		this.next = null;
	}
}

class LinkedListT{
	public LinkedListT(){
		head = null;
	}

	public void addAtBegin(int data){
		Node n = new Node(data);
		n.next = head;
		head = n;
	}

	public void reverseIterative(Node head){
		Node currNode = head;
		Node nextNode = null;
		Node prevNode = null;

		while(currNode != null){
			nextNode = currNode.next;
			currNode.next = prevNode;
			prevNode = currNode;
			currNode = nextNode;
		}// while ends
		head = prevNode;
		System.out.println("\n Reverse through iteration");
		display(head);
	}

	public void reverseRecursion(Node ptrOne,Node ptrTwo, Node prevNode){
		if(ptrTwo!=null){
				if(ptrTwo.next!=null){
					Node t1 = ptrTwo;
					Node t2 = ptrTwo.next;
					ptrOne.next = prevNode;
					prevNode = ptrOne;
					reverseRecursion(t1,t2, prevNode);
				}
				else{
					ptrTwo.next = ptrOne;
					ptrOne.next = prevNode;
					System.out.println("\n Reverse Through Recursion");
					display(ptrTwo);
				}
		}
		else if(ptrOne!=null){
			System.out.println("\n Reverse Through Recursion");
			display(ptrOne);
		}
	}// reverseRecursion ends

	public void display(Node head){
		Node cur = head;
		while(cur != null){
			System.out.print("->" + cur.data);
			cur = cur.data;
		}
	}// display ends
}