class BookNode:
    def _init_(self, book_id, title, author, status="Available"):
        self.book_id = book_id
        self.title = title
        self.author = author
        self.status = status
        self.next = None

class BookList:
    def _init_(self):
        self.head = None

    def insertBook(self, book_id, title, author, status="Available"):
        new_book = BookNode(book_id, title, author, status)
        if not self.head:
            self.head = new_book
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_book
        print(f" Book '{title}' added successfully!")

    def deleteBook(self, book_id):
        current = self.head
        prev = None
        while current and current.book_id != book_id:
            prev = current
            current = current.next
        if not current:
            print(" Book not found!")
            return
        if prev:
            prev.next = current.next
        else:
            self.head = current.next
        print(f" Book '{current.title}' deleted successfully!")

    def searchBook(self, book_id):
        current = self.head
        while current:
            if current.book_id == book_id:
                print(f" Found Book â ID: {current.book_id}, Title: {current.title}, Author: {current.author}, Status: {current.status}")
                return current
            current = current.next
        print(" Book not found!")
        return None

    def displayBooks(self):
        if not self.head:
            print(" No books available in the library.")
            return
        current = self.head
        print("\n Current Library Books:")
        while current:
            print(f"ID: {current.book_id}, Title: {current.title}, Author: {current.author}, Status: {current.status}")
            current = current.next
        print("-" * 50)
      
class TransactionStack:
    def _init_(self):
        self.stack = []

    def push(self, transaction):
        self.stack.append(transaction)

    def pop(self):
        if not self.isEmpty():
            return self.stack.pop()
        else:
            print("No transactions to undo.")
            return None

    def isEmpty(self):
        return len(self.stack) == 0

    def viewTransactions(self):
        if self.isEmpty():
            print("No transactions yet.")
        else:
            print("\nTransaction History:")
            for t in reversed(self.stack):
                print(t)
            print("-" * 50)

class LibrarySystem:
    def _init_(self):
        self.book_list = BookList()
        self.transactions = TransactionStack()

    def issueBook(self, book_id):
        book = self.book_list.searchBook(book_id)
        if book and book.status == "Available":
            book.status = "Issued"
            self.transactions.push(("Issue", book_id))
            print(f"Book '{book.title}' issued successfully.")
        elif book:
            print("Book is already issued.")

    def returnBook(self, book_id):
        book = self.book_list.searchBook(book_id)
        if book and book.status == "Issued":
            book.status = "Available"
            self.transactions.push(("Return", book_id))
            print(f"Book '{book.title}' returned successfully.")
        elif book:
            print("Book is already available.")

    def undoTransaction(self):
        last_transaction = self.transactions.pop()
        if not last_transaction:
            return
        action, book_id = last_transaction
        book = self.book_list.searchBook(book_id)
        if not book:
            return
        if action == "Issue":
            book.status = "Available"
            print(f"Undo: Book '{book.title}' marked as Available again.")
        elif action == "Return":
            book.status = "Issued"
            print(f"Undo: Book '{book.title}' marked as Issued again.")

    def viewTransactions(self):
        self.transactions.viewTransactions()


if _name_ == "_main_":
    system = LibrarySystem()

    # Inserting books
    system.book_list.insertBook(101, "The Alchemist", "Paulo Coelho")
    system.book_list.insertBook(102, "Atomic Habits", "James Clear")
    system.book_list.insertBook(103, "Rich Dad Poor Dad", "Robert Kiyosaki")

    # Display books
    system.book_list.displayBooks()

    # Issue and return operations
    system.issueBook(101)
    system.issueBook(102)
    system.returnBook(101)
    system.book_list.displayBooks()

    # Undo last transaction
    system.undoTransaction()
    system.book_list.displayBooks()

    # View transactions
    system.viewTransactions()
