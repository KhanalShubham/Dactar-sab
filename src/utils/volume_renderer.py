import plotly.graph_objects as go
import numpy as np
import logging

logger = logging.getLogger("VolumeRenderer")

def create_3d_isosurface(volume, spacing, threshold=0.4):
    """
    Generates a 3D Isosurface figure using Plotly's native voxel renderer.
    volume: (Z, Y, X)
    spacing: (Y_spacing, X_spacing, Z_spacing)
    """
    try:
        # Downsample for performance if needed
        z, y, x = volume.shape
        step = 1
        if z * y * x > 128**3:
            step = 2
            
        vol_ds = volume[::step, ::step, ::step]
        
        # Create coordinate grids
        zz, yy, xx = np.mgrid[0:z:step, 0:y:step, 0:x:step]
        
        # Scale coordinates by clinical spacing
        zz = zz * spacing[2]
        yy = yy * spacing[0]
        xx = xx * spacing[1]

        fig = go.Figure(data=go.Isosurface(
            x=xx.flatten(),
            y=yy.flatten(),
            z=zz.flatten(),
            value=vol_ds.flatten(),
            isomin=threshold,
            isomax=1.0,
            opacity=0.6,
            surface_count=3, 
            colorscale='Viridis',
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=False
        ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X (mm)", backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True),
                yaxis=dict(title="Y (mm)", backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True),
                zaxis=dict(title="Z (mm)", backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True),
                aspectmode='data'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        return fig
    except Exception as e:
        logger.error(f"3D Rendering failed: {e}")
        return None

def create_mpr_slice(volume, plane='axial', index=0, spacing=(1.0, 1.0, 1.0)):
    """
    Generates a 2D slice plot for MPR.
    plane: 'axial' (Z), 'sagittal' (X), 'coronal' (Y)
    """
    try:
        if plane == 'axial':
            slice_data = volume[index, :, :]
            ratio = spacing[1] / spacing[0]
            title = f"Axial Slice {index}"
        elif plane == 'sagittal':
            slice_data = volume[:, :, index]
            ratio = spacing[2] / spacing[0] # Z / Y
            title = f"Sagittal Slice {index}"
        elif plane == 'coronal':
            slice_data = volume[:, index, :]
            ratio = spacing[2] / spacing[1] # Z / X
            title = f"Coronal Slice {index}"
        else:
            return None

        fig = go.Figure(data=go.Heatmap(
            z=np.flipud(slice_data),
            colorscale='Gray',
            showscale=False,
            hoverinfo='none'
        ))
        
        # Adaptive aspect ratio: if volume is very thin, disable fixed ratio to allow visibility
        if plane != 'axial' and volume.shape[0] < 15:
            yaxis_config = dict(showgrid=False, zeroline=False, showticklabels=False)
        else:
            yaxis_config = dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=ratio)

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=14, color='white')),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=yaxis_config,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, b=10, t=40),
            height=400
        )
        return fig
    except Exception as e:
        logger.error(f"Slice plotting failed: {e}")
        return None


def create_3d_with_highlight(volume, spacing, threshold=0.3, highlight_z_fraction=None):
    """
    Renders a 3D Isosurface (Greys colorscale) and optionally adds a semi-transparent
    cyan horizontal plane + outline at a given fractional Z position.
    volume: (Z, Y, X)
    spacing: (Y_spacing, X_spacing, Z_spacing)
    highlight_z_fraction: float 0.0–1.0, or None for no highlight
    """
    try:
        z, y, x = volume.shape
        step = 1
        if z * y * x > 128**3:
            step = 2

        vol_ds = volume[::step, ::step, ::step]

        zz, yy, xx = np.mgrid[0:z:step, 0:y:step, 0:x:step]
        zz = zz * spacing[2]
        yy = yy * spacing[0]
        xx = xx * spacing[1]

        traces = [
            go.Isosurface(
                x=xx.flatten(),
                y=yy.flatten(),
                z=zz.flatten(),
                value=vol_ds.flatten(),
                isomin=threshold,
                isomax=1.0,
                opacity=0.55,
                surface_count=2,
                colorscale='Greys',
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False,
            )
        ]

        x_mm = (x - 1) * spacing[1]
        y_mm = (y - 1) * spacing[0]

        scene_camera = None
        if highlight_z_fraction is not None:
            z_mm = highlight_z_fraction * (z - 1) * spacing[2]

            # Semi-transparent cyan horizontal plane
            traces.append(go.Surface(
                x=[[0, x_mm], [0, x_mm]],
                y=[[0, 0], [y_mm, y_mm]],
                z=[[z_mm, z_mm], [z_mm, z_mm]],
                opacity=0.22,
                colorscale=[[0, '#0EA5E9'], [1, '#0EA5E9']],
                showscale=False,
            ))

            # Cyan rectangle outline around the plane
            traces.append(go.Scatter3d(
                x=[0, x_mm, x_mm, 0, 0],
                y=[0, 0, y_mm, y_mm, 0],
                z=[z_mm, z_mm, z_mm, z_mm, z_mm],
                mode='lines',
                line=dict(color='#0EA5E9', width=3),
                showlegend=False,
            ))

            t = highlight_z_fraction
            eye_z = 2.5 - t * 1.8
            scene_camera = dict(
                eye=dict(x=1.8, y=1.8, z=eye_z),
                center=dict(x=0, y=0, z=(t - 0.5) * 0.6),
            )

        fig = go.Figure(data=traces)

        scene_dict = dict(
            xaxis=dict(title="X (mm)", backgroundcolor="rgb(12,12,22)", gridcolor="gray", showbackground=True),
            yaxis=dict(title="Y (mm)", backgroundcolor="rgb(12,12,22)", gridcolor="gray", showbackground=True),
            zaxis=dict(title="Z (mm)", backgroundcolor="rgb(12,12,22)", gridcolor="gray", showbackground=True),
            aspectmode='data',
        )
        if scene_camera is not None:
            scene_dict['camera'] = scene_camera

        fig.update_layout(
            scene=scene_dict,
            paper_bgcolor='rgb(12,12,22)',
            plot_bgcolor='rgb(12,12,22)',
            margin=dict(l=0, r=0, b=0, t=0),
            height=460,
        )
        return fig
    except Exception as e:
        logger.error(f"3D highlight rendering failed: {e}")
        return None
