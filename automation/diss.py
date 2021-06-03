from discord.ext import tasks
import socket
import discord

class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # an attribute we can access from our task
        self.counter = 0

        # start the task to run in the background
        self.my_background_task.start()
        self.content = 'none'
        self.flag = 999
        self.pre = 'initial'
        self.pre_training = 'initial'

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    # async def connect_to_local(self):
    #     print('socket')
    #     self.s = socket.socket()
    #     host = socket.gethostname()
    #     port =12345
    #     self.s.bind((host, port))
    #     self.s.listen(3)
    #     print('waiting for connection')
    #     self.c, addr = self.s.accept()
    #     print('connected')


    @tasks.loop(seconds=10) # task runs every 60 seconds
    async def my_background_task(self):
        print('====')
        channel = self.get_channel(843253934252228651) # channel ID goes here


        with open('/home/ubuntu/TA3N/status.txt', 'r') as f:
            self.content = f.read()

        # self.content = self.c.recv(1024).decode()
        print(self.content)
        # training_content = self.content.split('\n')[0]
        things = self.content.split('\n')
        if len(things) == 3:
            epoch = things[1]
        else:
            if self.pre == self.content:
                return
            else:
                self.pre = self.content
                await channel.send(self.content)
                return
        print('pre:',self.pre)
        if self.content == self.pre:
            return
        elif epoch == '49' or epoch == '1' or epoch == '15':
            self.pre = self.content
            await channel.send(self.content)


    async def on_message(self, message):
        # don't respond to ourselves
        if message.author == self.user:
            return

        if message.content == 'show':
            await message.channel.send(self.content)


    @my_background_task.before_loop
    async def before_my_task(self):
        # await self.connect_to_local()
        await self.wait_until_ready() # wait until the bot logs in

client = MyClient()
client.run('ODQ5Mzk2MDgwNDM3NDkzODEw.YLajvg.QlJP8dZE4OdrglVa2-sVc4JraEQ')