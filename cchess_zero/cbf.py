import xmltodict

template = """<?xml version="1.0" encoding="UTF-8"?>
<ChineseChessRecord Version="1.0">
 <Head>
  <Name>{{name}}</Name>
  <URL />
  <From>{{from}}</From>
  <ContestType />
  <Contest />
  <Round>{{round}}</Round>
  <Group />
  <Table />
  <Date>{{date}}</Date>
  <Site>{{site}}</Site>
  <TimeRule />
  <Red>{{red}}</Red>
  <RedTeam>{{redteam}}</RedTeam>
  <RedTime />
  <RedRating />
  <Black>{{black}}</Black>
  <BlackTeam>{{blackteam}}</BlackTeam>
  <BlackTime />
  <BlackRating />
  <Referee />
  <Recorder />
  <Commentator />
  <CommentatorURL />
  <Creator />
  <CreatorURL />
  <DateCreated />
  <DateModified>{{datemodified}}</DateModified>
  <ECCO>D21</ECCO>
  <RecordType>1</RecordType>
  <RecordKind />
  <RecordResult>0</RecordResult>
  <ResultType />
  <FEN>rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1</FEN>
 </Head>
 <MoveList>
 <Move value="00-00" />
{{body}}
 </MoveList>
</ChineseChessRecord>"""
move_template = """  <Move value="{{move}}" />
"""
end_template = """  <Move value="{{move}}" end="1" />
"""
xdic = dict(zip('abcdefghi','012345678'))
ydic = dict(zip('9876543210','0123456789'))

xdic_rev = dict(zip('012345678','abcdefghi'))
ydic_rev = dict(zip('9876543210','0123456789'))

def cbf2move(onefile):
    out_move = []
    doc = xmltodict.parse(open(onefile,encoding='utf-8').read())
    fen = doc['ChineseChessRecord']["Head"]["FEN"]
    pgnfile = doc['ChineseChessRecord']["Head"]["From"]
    moves = [i["@value"] for i in  doc['ChineseChessRecord']['MoveList']["Move"] if i["@value"] != '00-00']
    for i in moves:
        x1,y1,x2,y2 = str(i[0]),str(i[1]),str(i[3]),str(i[4])
        out_move.append("{}{}{}{}".format(xdic_rev[x1],ydic_rev[y1],xdic_rev[x2],ydic_rev[y2]))
    return out_move
        
class CBF():
    def __init__(self,**meta):
        self.text = template
        for key in meta:
            val = meta[key]
            self.text = self.text.replace("{{" + key + "}}",val)
    def receive_moves(self,moves):
        self.body = ""
        for move in moves[:-1]:
            a,b,c,d = move
            move = "{}{}-{}{}".format(xdic[a],ydic[b],xdic[c],ydic[d])
            self.body += move_template.replace("{{move}}",move)
        a,b,c,d = moves[-1]
        move = "{}{}-{}{}".format(xdic[a],ydic[b],xdic[c],ydic[d])
        self.body += end_template.replace("{{move}}",move)
        self.text = self.text.replace("{{body}}",self.body)
    def dump(self,filename):
        with open(filename,'w',encoding='utf-8') as whdl:
            whdl.write(self.text)