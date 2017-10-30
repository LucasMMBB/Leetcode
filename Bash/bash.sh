# 195. Tenth Line
# Read from the file file.txt and output the tenth line to stdout.

# solution 1
awk 'NR'=='10' file.txt

# soution 2
sed -n 10p file.txt

# soutin 3
tail -n+10 file.txt|head-1