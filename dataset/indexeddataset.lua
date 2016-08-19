--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local IndexedDataset, Dataset = torch.class('tnt.IndexedDataset', 'tnt.Dataset', tnt)
local IndexedDatasetReader = torch.class('tnt.IndexedDatasetReader', tnt)
local IndexedDatasetWriter = torch.class('tnt.IndexedDatasetWriter', tnt)

-- dataset built on index reader
IndexedDataset.__init = argcheck{
   doc = [[
<a name="IndexedDataset">
#### tnt.IndexedDataset(@ARGP)
@ARGT

A `tnt.IndexedDataset()` is a data structure built upon (possibly several)
data archives containing a bunch of tensors of the same type.

See [tnt.IndexedDatasetWriter](#IndexedDatasetWriter) and
[tnt.IndexedDatasetReader](#IndexedDatasetReader) to see how to create and
read a single archive.

Purpose: large datasets (containing a lot of files) are often not very well
handled by filesystems (especially over network). `tnt.IndexedDataset`
provides a convenient and efficient way to bundle them into a single
archive file, associated with an indexed file.

If `path` is provided, then `fields` must be a Lua array (keys being
numbers), where values are string representing a filename prefix to a
(index,archive) pair.  In other word `path/field.{idx,bin}` must exist. The
i-th sample returned by this dataset will be a table containing each field
as key, and a tensor found at the corresponding archive at index i.

If `path` is not provided, then `fields` must be a Lua hash. Each key
represents sample fields and the corresponding value must be a table
containing the keys `idx` (for the index filename path) and `bin` (for the
archive filename path).

If provided (and positive), `maxload` limits the dataset size to the
specified size.

Archives and/or indexes can also be memory mapped with the `mmap` and
`mmapidx` flags.

]],
   noordered = true,
   {name='self', type='tnt.IndexedDataset'},
   {name='fields', type='table'},
   {name='path', type='string', opt=true},
   {name='maxload', type='number', opt=true},
   {name='mmap', type='boolean', default=false},
   {name='mmapidx', type='boolean', default=false},
   call =
      function(self, fields, path, maxload, mmap, mmapidx)
         self.__fields = {}
         if path then
            for _, fieldname in ipairs(fields) do
               assert(type(fieldname) == 'string', 'fields should be a list of strings (fieldnames)')
               table.insert(
                  self.__fields, {
                     name = fieldname,
                     idx = string.format('%s/%s.idx', path, fieldname),
                     bin = string.format('%s/%s.bin', path, fieldname)})
            end
         else
            for fieldname, paths in pairs(fields) do
               assert(
                  type(fieldname) == 'string'
                     and type(paths.bin) == 'string'
                     and type(paths.idx) == 'string',
                  'fields should be a hash of string (fieldname) -> table (idx=string, bin=string)')
               table.insert(self.__fields, {
                               name = fieldname,
                               idx = paths.idx,
                               bin = paths.bin})
            end
         end
         assert(#self.__fields > 0, 'fields should not be empty')
         local size
         for _, field in ipairs(self.__fields) do
            field.data = tnt.IndexedDatasetReader(field.idx, field.bin, mmap, mmapidx)
            if size then
               assert(field.data:size() == size, 'inconsistent index data size')
            else
               size = field.data:size()
            end
         end
         if maxload and maxload > 0 and maxload < size then
            size = maxload
         end
         self.__size = size
         print(string.format("| IndexedDataset: loaded %s with %d examples", path or '', size))
      end
}

IndexedDataset.size = argcheck{
   {name='self', type='tnt.IndexedDataset'},
   call =
      function(self)
         return self.__size
      end
}

IndexedDataset.get = argcheck{
   {name='self', type='tnt.IndexedDataset'},
   {name='idx', type='number'},
   call =
      function(self, idx)
         assert(idx >= 1 and idx <= self.__size, 'index out of bound')
         local sample = {}
         for _, field in ipairs(self.__fields) do
            sample[field.name] = field.data:get(idx)
         end
         return sample
      end
}

-- supported tensor types
local IndexedDatasetIndexTypes = {}
for code, type in ipairs{'byte', 'char', 'short', 'int', 'long', 'float', 'double', 'table'} do
   local Type = (type == 'table') and 'Char' or type:sub(1,1):upper() .. type:sub(2)
   IndexedDatasetIndexTypes[type] = {
      code = code,
      name = string.format('torch.%sTensor', Type),
      read = string.format('read%s', Type),
      write = string.format('write%s', Type),
      size = torch[string.format('%sStorage', Type)].elementSize(),
      storage = torch[string.format('%sStorage', Type)],
      tensor = torch[string.format('%sTensor', Type)]
   }
end

-- index reader helper function
local function readindex(self, indexfilename)
   local f = torch.DiskFile(indexfilename):binary()
   assert(f:readLong() == 0x584449544E54, "unrecognized index format")
   assert(f:readLong() == 1, "unsupported format version")
   local code = f:readLong()
   for typename, type in pairs(IndexedDatasetIndexTypes) do
      if type.code == code then
         self.type = type
         self.typename = typename
      end
   end
   assert(self.type, "unrecognized type")
   assert(f:readLong() == self.type.size, "type size do not match")
   self.N = f:readLong()
   self.S = f:readLong()
   self.dimoffsets = torch.LongTensor(f:readLong(self.N+1))
   self.datoffsets = torch.LongTensor(f:readLong(self.N+1))
   self.sizes = torch.LongTensor(f:readLong(self.S))
   f:close()
end

-- index writer
IndexedDatasetWriter.__init = argcheck{
   doc = [[
<a name="IndexedDatasetWriter">
##### tnt.IndexedDatasetWriter(@ARGP)
@ARGT

Creates a (archive,index) file pair. The archive will contain tensors of the same specified `type`.

`type` must be a string chosen in {`byte`, `char`, `short`, `int`, `long`, `float`, `double` or `table`}.

`indexfilename` is the full path to the index file to be created.
`datafilename` is the full path to the data archive file to be created.

Tensors are added to the archive with [add()](#IndexedDataset.add).

Note that you *must* call [close()](#IndexedDataset.close) to ensure all
data is written on disk and to create the index file.

The type `table` is special: data will be stored into a CharTensor,
serialized from a Lua table object. IndexedDatasetReader will then
deserialize the CharTensor into a table at read time. This allows storing
heterogenous data easily into an IndexedDataset.

]],
   {name='self', type='tnt.IndexedDatasetWriter'},
   {name='indexfilename', type='string'},
   {name='datafilename', type='string'},
   {name='type', type='string'},
   call =
      function(self, indexfilename, datafilename, type)
         self.BLOCKSZ = 1024
         self.indexfilename = indexfilename
         self.datafilename = datafilename
         assert(IndexedDatasetIndexTypes[type], 'invalid type (byte, char, short, int, long, float, double or table expected)')
         self.dimoffsets = torch.LongTensor(self.BLOCKSZ)
         self.datoffsets = torch.LongTensor(self.BLOCKSZ)
         self.sizes = torch.LongTensor(self.BLOCKSZ)
         self.N = 0
         self.S = 0
         self.dimoffsets[1] = 0
         self.datoffsets[1] = 0
         self.type = IndexedDatasetIndexTypes[type]
         self.datafile = torch.DiskFile(datafilename, 'w'):binary()
      end
}

-- append mode
IndexedDatasetWriter.__init = argcheck{
   doc = [[
##### tnt.IndexedDatasetWriter(@ARGP)
@ARGT

Opens an existing (archive,index) file pair for appending. The tensor type is inferred from the provided
index file.

`indexfilename` is the full path to the index file to be opened.
`datafilename` is the full path to the data archive file to be opened.

]],
   {name='self', type='tnt.IndexedDatasetWriter'},
   {name='indexfilename', type='string'},
   {name='datafilename', type='string'},
   overload = IndexedDatasetWriter.__init,
   call =
      function(self, indexfilename, datafilename)
         self.BLOCKSZ = 1024
         self.indexfilename = indexfilename
         self.datafilename = datafilename
         readindex(self, indexfilename)
         self.datafile = torch.DiskFile(datafilename, 'rw'):binary()
         self.datafile:seekEnd()
      end
}

IndexedDatasetWriter.add = argcheck{
   doc = [[
<a name="IndexedDatasetWriter.add">
###### tnt.IndexedDatasetWriter.add(@ARGP)
@ARGT

Add a tensor to the archive and record its index position. The tensor type must of the same type
than the one specified at the creation of the `tnt.IndexedDatasetWriter`.
]],
   {name='self', type='tnt.IndexedDatasetWriter'},
   {name='tensor', type='torch.*Tensor'},
   call =
      function(self, tensor)
         assert(torch.typename(tensor) == self.type.name, 'invalid tensor type')
         local size = tensor:size()
         local dim = size:size()
         local N = self.N + 1
         local S = self.S + dim
         if self.dimoffsets:size(1) < N+1 then -- +1 for the first 0 value
            self.dimoffsets:resize(N+self.BLOCKSZ)
            self.datoffsets:resize(N+self.BLOCKSZ)
         end
         if self.sizes:size(1) < S then
            self.sizes:resize(S+self.BLOCKSZ)
         end
         self.dimoffsets[N+1] = self.dimoffsets[N] + dim
         self.datoffsets[N+1] = self.datoffsets[N] + tensor:nElement()
         if dim > 0 then
            self.sizes:narrow(1, self.S+1, dim):copy(torch.LongTensor(size))
         end
         self.N = N
         self.S = S
         if tensor:nElement() > 0 then
            self.datafile[self.type.write](self.datafile, tensor:clone():storage())
         end
      end
}

IndexedDatasetWriter.add = argcheck{
   doc = [[
###### tnt.IndexedDatasetWriter.add(@ARGP)
@ARGT

Convenience method which given a `filename` will open the corresponding
file in `binary` mode, and reads all data in there as if it was of the type
specified at the `tnt.IndexedDatasetWriter` construction.  A corresponding
tensor is then added to the archive/index pair.
]],
   {name='self', type='tnt.IndexedDatasetWriter'},
   {name='filename', type='string'},
   overload = IndexedDatasetWriter.add,
   call =
      function(self, filename)
         local f = torch.DiskFile(filename):binary()
         f:seekEnd()
         local sz = f:position()-1
         f:seek(1)
         local storage = f[self.type.read](f, sz/self.type.size)
         f:close()
         self:add(self.type.tensor(storage))
      end
}

IndexedDatasetWriter.add = argcheck{
   doc = [[
###### tnt.IndexedDatasetWriter.add(@ARGP)
@ARGT

Convenience method only available for `table` type IndexedDataset.
The table will be serialized into a CharTensor.

]],
   {name='self', type='tnt.IndexedDatasetWriter'},
   {name='table', type='table'},
   nonamed = true, -- ambiguity possible with table arg
   overload = IndexedDatasetWriter.add,
   call =
      function(self, tbl)
         assert(
            self.type == IndexedDatasetIndexTypes.table
               or self.type == IndexedDatasetIndexTypes.char,
            'table convenience method is only available for "table" or "char"-based datasets')
         tbl = torch.CharTensor(torch.serializeToStorage(tbl))
         self:add(tbl)
      end
}

IndexedDatasetWriter.close = argcheck{
   doc = [[
###### tnt.IndexedDatasetWriter.add(@ARGP)
@ARGT

Finalize the index, and Close the archive/index filename pair. This method
must be called to ensure the index is written and all the archive data is
flushed on disk.
]],
   {name='self', type='tnt.IndexedDatasetWriter'},
   call =
      function(self)
         local f = torch.DiskFile(self.indexfilename, 'w'):binary()
         f:writeLong(0x584449544E54) -- magic number
         f:writeLong(1) -- version
         f:writeLong(self.type.code) -- type code
         f:writeLong(self.type.size)
         f:writeLong(self.N)
         f:writeLong(self.S)

         -- resize properly underlying storages
         self.dimoffsets = torch.LongTensor( self.dimoffsets:storage():resize(self.N+1) )
         self.datoffsets = torch.LongTensor( self.datoffsets:storage():resize(self.N+1) )
         self.sizes = torch.LongTensor( self.sizes:storage():resize(self.S) )

         -- write index on disk
         f:writeLong(self.dimoffsets:storage())
         f:writeLong(self.datoffsets:storage())
         f:writeLong(self.sizes:storage())
         f:close()
         self.datafile:close()
      end
}

-- helper function that updates the meta table when data is on file:
local function updatemetatable(data, datafilename)
   local data_mt = {}
   local f = torch.DiskFile(datafilename):binary():noBuffer()
   function data_mt:narrow(dim, offset, size)
      f:seek((offset - 1) * self.type.size + 1)
      return self.type.tensor(f[self.type.read](f, size))
   end
   setmetatable(data, {__index = data_mt})
   return data
end

-- index reader
IndexedDatasetReader.__init = argcheck{
   doc = [[
<a name="IndexedDatasetReader">
##### tnt.IndexedDatasetReader(@ARGP)
@ARGT

Reads an archive/index pair previously created by
[tnt.IndexedDatasetWriter](#IndexedDatasetWriter).

`indexfilename` is the full path to the index file.
`datafilename` is the full path to the archive file.

Memory mapping can be specified for both the archive and index through the
optional `mmap` and `mmapidx` flags.

]],
   {name='self', type='tnt.IndexedDatasetReader'},
   {name='indexfilename', type='string'},
   {name='datafilename', type='string'},
   {name='mmap', type='boolean', default=false},
   {name='mmapidx', type='boolean', default=false},
   call =
      function(self, indexfilename, datafilename, mmap, mmapidx)
         self.indexfilename = indexfilename
         self.datafilename = datafilename

         if mmapidx then -- memory mapped index
            local idx = torch.LongStorage(indexfilename)
            local offset = 1
            assert(idx[offset] == 0x584449544E54, "unrecognized index format")
            offset = offset + 1
            assert(idx[offset] == 1, "unsupported format version")
            offset = offset + 1
            local code = idx[offset]
            offset = offset + 1
            for typename, type in pairs(IndexedDatasetIndexTypes) do
               if type.code == code then
                  self.type = type
                  self.typename = typename
               end
            end
            assert(self.type, "unrecognized type")
            assert(idx[offset] == self.type.size, "type size do not match")
            offset = offset + 1
            self.N = idx[offset]
            offset = offset + 1
            self.S = idx[offset]
            offset = offset + 1
            self.dimoffsets = torch.LongTensor(idx, offset, self.N+1)
            offset = offset + self.N+1
            self.datoffsets = torch.LongTensor(idx, offset, self.N+1)
            offset = offset + self.N+1
            self.sizes = torch.LongTensor(idx, offset, self.S)
         else -- index on file
            readindex(self, indexfilename)
         end

         if mmap then -- memory mapped data
            self.data = self.type.tensor(self.type.storage(datafilename))
         else -- data on file
            local data = {type=self.type}
            self.data = updatemetatable(data, datafilename)
         end
      end
}

function IndexedDatasetReader:__write(file)
   local obj = {}
   for k,v in pairs(self) do obj[k] = v end
   obj.type = nil
   if type(self.data) == 'table' then obj.data.type = nil end
   file:writeObject(obj)
   if type(self.data) == 'table' then obj.data.type = self.type end
end

function IndexedDatasetReader:__read(file)
   for k,v in pairs(file:readObject()) do
      self[k] = v
   end
   self.type = IndexedDatasetIndexTypes[self.typename]
   if type(self.data) == 'table' then
      self.data.type = self.type
      updatemetatable(self.data, self.datafilename)
   end
end

IndexedDatasetReader.size = argcheck{
   doc = [[
<a name="IndexedDatasetReader.size">
###### tnt.IndexedDatasetReader.size(@ARGP)

Returns the number of tensors present in the archive.
]],
   {name='self', type='tnt.IndexedDatasetReader'},
   call =
      function(self)
         return self.N
      end
}

IndexedDatasetReader.get = argcheck{
   doc = [[
<a name="IndexedDatasetReader.get">
###### tnt.IndexedDatasetReader.get(@ARGP)

Returns the tensor at the specified `index` in the archive.
]],
   {name='self', type='tnt.IndexedDatasetReader'},
   {name='index', type='number'},
   call =
      function(self, index)
         assert(index > 0 and index <= self.N, 'index out of range')
         local ndim = self.dimoffsets[index+1]-self.dimoffsets[index]
         if ndim == 0 then
            return self.type.tensor()
         end
         local size = self.sizes:narrow(
            1,
            self.dimoffsets[index]+1,
            ndim
         )
         size = size:clone():storage()
         local data = self.data:narrow(
            1,
            self.datoffsets[index]+1,
            self.datoffsets[index+1]-self.datoffsets[index]
         ):view(size)
         if self.type == IndexedDatasetIndexTypes.table then
            return torch.deserializeFromStorage(data)
         else
            return data:clone()
         end
      end
}
