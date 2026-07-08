import wx.lib.activex
 
# Only do this once in the notebook
app = wx.App(False)
cam_frame = wx.Frame(None, title="DataRay OCX host", size=(200, 100))
panel = wx.Panel(cam_frame)
 
gd_activex = wx.lib.activex.ActiveXCtrl(
    panel,
    "DATARAYOCX.GetDataCtrl.1",
    pos=(0, 0),
    size=(10, 10)
)
 
gd_ctrl = gd_activex.ctrl
 
gd_ctrl.StartDriver()
gd_ctrl.SetCurrentDevice(5)      # WinCam
gd_ctrl.SetLiveRecallState(0)    # Live mode
gd_ctrl.StartDevice()
 
cam_frame.Show()
print("Camera initialized")  from time import sleep