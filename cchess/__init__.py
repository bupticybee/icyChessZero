# -*- coding: utf-8 -*-

'''
Copyright (C) 2014  walker li <walker8088@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from cchess.piece import *
from cchess.board import *
from cchess.game import  *
from cchess.move import  *
from cchess.ucci  import *
from cchess.reader_xqf import read_from_xqf
from cchess.reader_cbf import read_from_cbf
from cchess.reader_pgn import read_from_pgn
#from cchess.reader_dhtml import read_from_dhtml 
from cchess.exception import *