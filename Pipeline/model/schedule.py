import dill
import pandas as pd
import tzlocal
from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler(timezone=tzlocal.get_localzone_name())

df = pd.read_csv('data/balanced_df.csv')
with open('event_pipe.pkl', 'rb') as file:
    model = dill.load(file)


@sched.scheduled_job('cron', second='*/5')
def on_time():
    data = df.sample(5)
    data['predicted_value'] = model['model'].predict(data)
    print(data[['client_id', 'event_action', 'predicted_value']])


if __name__ == '__main__':
    sched.start()
