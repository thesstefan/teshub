import './LegendItem.css'

type LegendItemProps = {
    name: string;
    color: string;
};

function LegendItem(props: LegendItemProps) {
    const { name } = props;
    const colorStyle = { backgroundColor: props.color }; 

    return <div className="legend-item">
        <div className="legend-item-color" style={colorStyle}/>
        <span className="legend-item-name">
            {name}
        </span>
    </div>;
}

export default LegendItem;
