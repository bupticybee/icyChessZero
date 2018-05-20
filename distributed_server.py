import tornado.ioloop
import tornado.web
import argparse
import os
datadir = 'data/distributed'

parser = argparse.ArgumentParser(description="mcts self play script") 
parser.add_argument('--verbose', '-v', help='verbose mode',type=bool,default=False)
parser.add_argument('--datadir', '-d' ,type=str,help="data dir to store chess plays",default=datadir)
args = parser.parse_args()
datadir = args.datadir


class TestHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("OK")

class ChessSubmitHandler(tornado.web.RequestHandler):
    def post(self):
        name = self.get_argument("name") 
        content = self.get_argument("content") 
        print("receive {}".format(name))
        if args.verbose == True:
            print(name,content)
        with open(os.path.join(datadir,name),'w',encoding='utf-8') as whdl:
            whdl.write(content)
        self.write("OK")

def make_app():
    return tornado.web.Application([
        (r"/test", TestHandler),
        (r"/submit_chess", ChessSubmitHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(10086)
    tornado.ioloop.IOLoop.current().start()
