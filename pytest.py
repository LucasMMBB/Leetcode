nums = [4, 3, 2, 7, 8, 2, 3, 1]
ans = []
for i in range(len(nums)):
    index = abs(nums[i]) - 1
    print index
    if(nums[index] > 0):
        nums[index] = -nums[index]

for i in range(len(nums)):
    if nums[i] > 0:
        ans.append(i + 1)

print ans