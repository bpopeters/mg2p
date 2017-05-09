function split(str)
    tbl = {}
    for c in str:gmatch('([^,]+)') do
         table.insert(tbl, c)
    end
    return tbl
end

big_tbl = {}
for line in io.lines(arg[1]) do
    table.insert(big_tbl, split(line))
end
vocab_matrix = torch.Tensor(big_tbl)
torch.save(arg[2], vocab_matrix)
