package = "torchnet"
version = "scm-1"

source = {
   url = "git://github.com/torchnet/torchnet.git"
}

description = {
   summary = "Torch on steroids",
   detailed = [[
   Various abstractions for torch7.
   ]],
   homepage = "https://github.com/torchnet/torchnet",
   license = "BSD"
}

dependencies = {
   "lua >= 5.1",
   "torch >= 7.0",
   "nn >= 1.0",
   "argcheck >= 1.0",
   "threads >= 1.0",
   "md5 >= 1.0",
   "luafilesystem >= 1.0",
   "luasocket >= 1.0",
   "optim >= 1.0",
   "tds >= 1.0",
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      LUA_PATH="$(LUADIR)",
      LUA_CPATH="$(LIBDIR)"
   }
}
