import parl

@parl.remote_class
class Actor(object):
  def hello_world(self):
      print("Hello world.")

  def add(self, a, b):
      return a + b

# connect to master node in cluster
parl.connect("localhost:6006")

actor = Actor()
actor.hello_world()# because we are running on the cluster so this line will print nothing